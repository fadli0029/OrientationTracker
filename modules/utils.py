# -------------------------------------------------------------------------
# Author: Muhammad Fadli Alim Arsani
# Email: fadlialim0029[at]gmail.com
# File: utils.py
# Description: This file contains utility functions for reading data and
#              saving the plot of the optimized, observed, and IMU-measured
#              acceleration data, and the RPY angles of the optimized
#              quaternion, RPY angles of the quaternions from motion model,
#              and RPY angles from VICON Motion Capture System.
# Misc: This is also part of one of the projects in the course
#       "Sensing and Estimation in Robotics" taught by Prof. Nikolay
#       Atanasov @UC San Diego.
#       https://natanaso.github.io/ece276a/index.html
# -------------------------------------------------------------------------

import os
import sys
import pickle
import numpy as np
import jax.numpy as jnp
import transforms3d as tf3d
import matplotlib.pyplot as plt

def read_data(fname):
    """
    Read the data from the file and return it

    Args:
        fname (str): file name

    Returns:
        d: data
    """
    d = []
    with open(fname, 'rb') as f:
        if sys.version_info[0] < 3:
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='latin1') # needed for python 3
    return d

def read_imu_data(dataset, path='data/'):
    """
    Read the imu data from the dataset and return it

    Args:
        dataset (int): dataset number
        path (str): path to the data folder

    Returns:
        raw_imu_data: raw imu data of shape (N, 7)
    """
    dataset = str(dataset)

    imu_file = path + "imu/imuRaw" + dataset + ".p"

    # Reshape raw imu data to usual convention (in ML/Robotics)
    raw_imu_data = read_data(imu_file)
    raw_imu_data = np.hstack(
            (raw_imu_data['vals'].T, raw_imu_data['ts'].reshape(-1,1))
        )
    return raw_imu_data

def read_vicon_data(dataset, path='data/'):
    """
    Read the vicon data from the dataset and return it

    Args:
        dataset (int): dataset number
        path (str): path to the data folder

    Returns:
        vicon_data: a dictionary with keys 'rots' and 'ts',
        such that 'rots' is a numpy array of shape (N, 3, 3)
        and 'ts' is a numpy array of shape (N,)
    """
    dataset = str(dataset)
    vicon_file = path + "vicon/viconRot" + dataset + ".p"

    # Reshape vicon data to usual convention (in ML/Robotics)
    vicon_data = read_data(vicon_file)
    vicon_data['rots'] = np.transpose(vicon_data['rots'], (2, 0, 1))
    vicon_data['ts'] = vicon_data['ts'].reshape(-1,)
    return vicon_data

def read_camera_data(dataset, path='data/'):
    """
    Read the camera data from the dataset and return it

    Args:
        dataset (int): dataset number
        path (str): path to the data folder

    Returns:
        camera_data: a dictionary with keys 'ts' and 'images',
        such that 'ts' is a numpy array of shape (N,)
        and 'images' is a numpy array of shape (N, H, W, C)
    """
    dataset = str(dataset)
    camera_file = path + "cam/cam" + dataset + ".p"
    camera_data = read_data(camera_file)
    camera_data['cam'] = np.transpose(camera_data['cam'], (3, 0, 1, 2))
    camera_data['ts'] = camera_data['ts'].reshape(-1,)
    return camera_data

def read_dataset(dataset, path='data/', data_name='all'):
    """
    Read the dataset and return the raw imu data and vicon data

    Args:
        dataset (int): dataset number
        path (str): path to the data folder

    Returns:
        raw_imu_data: raw imu data of shape (N, 7)
        vicon_data: a dictionary with keys 'rots' and 'ts',
                    where 'rots' is a numpy array of shape (N, 3, 3)
                    and 'ts' is a numpy array of shape (N,)
    """
    valid_data_names = ['imu', 'vicon', 'camera', 'all']
    assert data_name in valid_data_names, 'Invalid data name'

    dataset = str(dataset)
    if data_name != 'all':
        if data_name == 'imu':
            return read_imu_data(dataset, path=path)
        elif data_name == 'vicon':
            return read_vicon_data(dataset, path=path)
        elif data_name == 'camera':
            return read_camera_data(dataset, path=path)
    elif data_name == 'all':
        return read_imu_data(dataset, path=path),\
               read_vicon_data(dataset, path=path)

def check_files_exist(datasets, results_folder):
    files = os.listdir(results_folder)
    files_exist = {dataset: False for dataset in datasets}
    for file in files:
        if file.endswith('.npy'):
            try:
                d = int(file.split('_')[2])
                if d in files_exist:
                    files_exist[d] = True
            except ValueError:
                continue
    return files_exist

def save_results(data: dict, f: str, folder_path: str, tr):
    """
    Save optimized quaternions, motion model quaternions, estimated
    accelerations, observed accelerations, and costs record to a file.

    Args:
        data: a dictionary with dataset number as the key and values
        is optimized quaternion 'q_optim' for example
        filename: file name
    """
    trackers = {
        'pgd': 'PGD',
        'ekf': 'EKF',
    }

    # If folder does not exist, create it. But strip the last '/' first
    if not os.path.exists(folder_path[:-1]):
        os.makedirs(folder_path[:-1])
    for key, val in data.items():
        dataset = str(key)
        filename = folder_path + f + "_" + str(key) + "_" + trackers[tr] + '.npy'
        jnp.save(filename, val)

def load_results(filename: str):
    """
    Load optimized quaternions, motion model quaternions, estimated
    accelerations, observed accelerations, and costs record from a file.

    Args:
        filename: file name
    """
    data = jnp.load(filename)
    return data

def quat2rot(q):
    assert q.shape[-1] == 4, "Input must have shape (..., 4)"
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)

    # use tf3d to convert quaternion to rotation matrix
    R = jnp.zeros((q.shape[0], 3, 3), dtype=jnp.float64)
    for i in range(q.shape[0]):
        R = R.at[i].set(jnp.array(tf3d.quaternions.quat2mat(q[i]), dtype=jnp.float64))
    return R

def save_plot(
    tr,
    q_optim,
    q_motion,
    a_optims,
    a_obsrv,
    accs_imu,
    dataset,
    save_image_folder,
    vicon_data=None,
    plot_model=False,
):
    """
    Save the plot of the optimized, observed, and IMU-measured acceleration
    data, and the RPY angles of the optimized quaternion, RPY angles of the
    quaternions from motion model, and RPY angles from VICON Motion Capture
    System.

    Args:
        tr: tracker name
        q_optim: optimized quaternion
        q_motion: quaternion from motion model
        a_estims: estimated acceleration
        a_obsrv: observed acceleration
        accs_imu: IMU-measured acceleration
        dataset: dataset number
        save_image_folder: folder to save the image
        vicon_data: a dictionary with keys 'rots' and 'ts',

    Returns:
        None
    """
    # First check if save_image_folder exists; if not, create it
    if not os.path.exists(save_image_folder):
        os.makedirs(save_image_folder)

    trackers = {
        'pgd': 'PGD',
        'ekf': 'EKF',
    }

    filename = save_image_folder + 'dataset_' + str(dataset) + '_' + trackers[tr] + '.png'
    ts = np.array(list(range(len(q_optim))))
    fig, axs = plt.subplots(3, 2, figsize=(20, 10))


    # Plotting the acceleration data
    axs[0, 0].plot(ts, a_optims[:, 0],  label='Optimized (Ax), ' + trackers[tr], color='r')
    axs[0, 0].plot(ts, accs_imu[1:, 0], label='IMU (Ax)',         color='b')
    axs[1, 0].plot(ts, a_optims[:, 1],  label='Optimized (Ay), ' + trackers[tr], color='r')
    axs[1, 0].plot(ts, accs_imu[1:, 1], label='IMU (Ay)',         color='b')
    axs[2, 0].plot(ts, a_optims[:, 2],  label='Optimized (Az), ' + trackers[tr], color='r')
    axs[2, 0].plot(ts, accs_imu[1:, 2], label='IMU (Az)',         color='b')

    if plot_model:
        axs[0, 0].plot(ts, a_obsrv[1:, 0],  label='Obsv. model (Ax)', color='g')
        axs[1, 0].plot(ts, a_obsrv[1:, 1],  label='Obsv. model (Ay)', color='g')
        axs[2, 0].plot(ts, a_obsrv[1:, 2],  label='Obsv. model (Az)', color='g')

    # Calculating Euler angles
    eulers_q_optim = np.array(euler(q_optim))
    eulers_q_motion = np.array(euler(q_motion))
    if vicon_data is not None:
        eulers_vicon = np.array(euler(vicon_data))
        eulers_vicon = eulers_vicon[1:]

    # Plotting the Euler angles
    axs[0, 1].plot(eulers_q_optim[:, 0],  label='Optimized (Roll), ' + trackers[tr], color='r')
    axs[1, 1].plot(eulers_q_optim[:, 1],  label='Optimized (Pitch), ' + trackers[tr], color='r')
    axs[2, 1].plot(eulers_q_optim[:, 2],  label='Optimized (Yaw), ' + trackers[tr], color='r')

    if plot_model:
        axs[0, 1].plot(eulers_q_motion[:, 0], label='Motion model (Roll)', color='g')
        axs[1, 1].plot(eulers_q_motion[:, 1], label='Motion model (Pitch)',color='g')
        axs[2, 1].plot(eulers_q_motion[:, 2], label='Motion model (Yaw)',  color='g')

    if vicon_data is not None:
        axs[0, 1].plot(eulers_vicon[:, 0],    label='Vicon (Roll)',        color='b')
        axs[1, 1].plot(eulers_vicon[:, 1],    label='Vicon (Pitch)',       color='b')
        axs[2, 1].plot(eulers_vicon[:, 2],    label='Vicon (Yaw)',         color='b')

    # Setting titles for each subplot
    axs[0, 0].set_title('Ax')
    axs[1, 0].set_title('Ay')
    axs[2, 0].set_title('Az')
    axs[0, 1].set_title('Roll')
    axs[1, 1].set_title('Pitch')
    axs[2, 1].set_title('Yaw')

    # Adding legends to each subplot
    axs[0, 0].legend()
    axs[1, 0].legend()
    axs[2, 0].legend()
    axs[0, 1].legend()
    axs[1, 1].legend()
    axs[2, 1].legend()

    fig.savefig(filename, bbox_inches='tight')

def euler(rot):
    """
    Convert rotation matrix or quaternion to euler angles.

    Args:
        rot: rotation matrix or quaternion, shape (N, 3, 3) or (N, 4)

    Returns:
        eulers: euler angles, shape (N, 3)
    """
    if type(rot) == dict:
        rot = rot['rots']
        return np.array([tf3d.euler.mat2euler(rot[i]) for i in range(rot.shape[0])])
    return np.array([tf3d.euler.quat2euler(q) for q in rot])
