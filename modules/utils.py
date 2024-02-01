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
    valid_data_names = ['imu', 'vicon', 'all']
    assert data_name in valid_data_names, 'Invalid data name'

    dataset = str(dataset)
    if data_name != 'all':
        if data_name == 'imu':
            return read_imu_data(dataset, path=path)
        elif data_name == 'vicon':
            return read_vicon_data(dataset, path=path)
    elif data_name == 'all':
        return read_imu_data(dataset, path=path),\
               read_vicon_data(dataset, path=path)

def save_plot(
    q_optim,
    q_motion,
    a_estims,
    a_obsrv,
    vicon_data,
    accs_imu,
    dataset,
    save_image_folder
):
    """
    Save the plot of the optimized, observed, and IMU-measured acceleration
    data, and the RPY angles of the optimized quaternion, RPY angles of the
    quaternions from motion model, and RPY angles from VICON Motion Capture
    System.

    Args:
        q_optim: optimized quaternion
        q_motion: quaternion from motion model
        a_estims: estimated acceleration
        a_obsrv: observed acceleration
        vicon_data: a dictionary with keys 'rots' and 'ts',
        accs_imu: IMU-measured acceleration
        dataset: dataset number
        save_image_folder: folder to save the image

    Returns:
        None
    """
    # First check if save_image_folder exists; if not, create it
    if not os.path.exists(save_image_folder):
        os.makedirs(save_image_folder)

    filename = save_image_folder + 'dataset_' + str(dataset) + '.png'
    ts = np.array(list(range(len(q_optim))))
    fig, axs = plt.subplots(3, 2, figsize=(20, 10))

    # Plotting the acceleration data
    axs[0, 0].plot(ts, a_estims[:, 0],  label='Optimized (Ax)',   color='r')
    axs[0, 0].plot(ts, a_obsrv[1:, 0],  label='Obsv. model (Ax)', color='g')
    axs[0, 0].plot(ts, accs_imu[1:, 0], label='IMU (Ax)',         color='b')
    axs[1, 0].plot(ts, a_estims[:, 1],  label='Optimized (Ay)',   color='r')
    axs[1, 0].plot(ts, a_obsrv[1:, 1],  label='Obsv. model (Ay)', color='g')
    axs[1, 0].plot(ts, accs_imu[1:, 1], label='IMU (Ay)',         color='b')
    axs[2, 0].plot(ts, a_estims[:, 2],  label='Optimized (Az)',   color='r')
    axs[2, 0].plot(ts, a_obsrv[1:, 2],  label='Obsv. model (Az)', color='g')
    axs[2, 0].plot(ts, accs_imu[1:, 2], label='IMU (Az)',         color='b')

    # Calculating Euler angles
    eulers_q_optim = np.array(euler(q_optim))
    eulers_q_motion = np.array(euler(q_motion))
    eulers_vicon = np.array(euler(vicon_data))

    # Since we estimated from 1:T, we need to slice q_motion and vicon_data
    # to match the size of q_optim
    q_motion = q_motion[1:]
    eulers_vicon = eulers_vicon[1:]

    # Plotting the Euler angles
    axs[0, 1].plot(eulers_q_optim[:, 0],  label='Optimized (Roll)',    color='r')
    axs[0, 1].plot(eulers_q_motion[:, 0], label='Motion model (Roll)', color='g')
    axs[0, 1].plot(eulers_vicon[:, 0],    label='Vicon (Roll)',        color='b')
    axs[1, 1].plot(eulers_q_optim[:, 1],  label='Optimized (Pitch)',   color='r')
    axs[1, 1].plot(eulers_q_motion[:, 1], label='Motion model (Pitch)',color='g')
    axs[1, 1].plot(eulers_vicon[:, 1],    label='Vicon (Pitch)',       color='b')
    axs[2, 1].plot(eulers_q_optim[:, 2],  label='Optimized (Yaw)',     color='r')
    axs[2, 1].plot(eulers_q_motion[:, 2], label='Motion model (Yaw)',  color='g')
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
