import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
from modules.preprocessing import load_all_vicon_datasets
import transforms3d as tf3d
from tqdm import tqdm

def euler(rot):
    if type(rot) == dict:
        rot = rot['rots']
        return np.array([tf3d.euler.mat2euler(rot[i]) for i in range(rot.shape[0])])
    return np.array([tf3d.euler.quat2euler(q) for q in rot])

def plot(e_vicon, e_pgd, e_kf, dataset, plot_folder):

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    fig, ax = plt.subplots(3, 1, figsize=(20, 10))
    ax[0].plot(e_vicon[:, 0], label="Vicon", color="g")
    ax[0].plot(e_pgd[:, 0], label="PGD", color="r")
    ax[0].plot(e_kf[:, 0], label="KF", color="b")
    ax[0].set_title(f"Roll - {dataset}")
    ax[0].legend()

    ax[1].plot(e_vicon[:, 1], label="Vicon", color="g")
    ax[1].plot(e_pgd[:, 1], label="PGD", color="r")
    ax[1].plot(e_kf[:, 1], label="KF", color="b")
    ax[1].set_title(f"Pitch - {dataset}")
    ax[1].legend()

    ax[2].plot(e_vicon[:, 2], label="Vicon", color="g")
    ax[2].plot(e_pgd[:, 2], label="PGD", color="r")
    ax[2].plot(e_kf[:, 2], label="KF", color="b")
    ax[2].set_title(f"Yaw - {dataset}")
    ax[2].legend()

    fname = "pgd_kf_vicon"
    plt.savefig(f"{plot_folder}/{fname}_{dataset}.png")

def main():
    parser = argparse.ArgumentParser(description="Plot results for comparison")
    parser.add_argument("--q_path", type=str, default="results/", help="Folder where the results are stored.")
    parser.add_argument("--plot_folder", type=str, default="plot_images/", help="Folder to save the plots.")
    parser.add_argument("--datasets", nargs="+", type=int, help="List of datasets to train and test the algorithm.")
    parser.add_argument("--path_to_datasets", type=str, default="data/trainset/", help="Path to the datasets.")

    args = parser.parse_args()
    q_path = args.q_path
    plot_folder = args.plot_folder
    datasets = args.datasets

    data = {}
    for dataset in datasets:
        data[dataset] = {}

        fname = f"{q_path}/q_optim_{dataset}"
        data[dataset]["pgd"] = np.load(f"{fname}_PGD.npy")
        data[dataset]["kf"] = np.load(f"{fname}_KF.npy")

    vicon_datasets = load_all_vicon_datasets(
        args.path_to_datasets, datasets
    )

    # use tqdm
    for dataset in tqdm(datasets):
        vicon = vicon_datasets[dataset]
        q_pgd = data[dataset]["pgd"]
        q_kf = data[dataset]["kf"]

        e_vicon = euler(vicon)
        e_pgd = euler(q_pgd)
        e_kf = euler(q_kf)

        plot(e_vicon, e_pgd, e_kf, dataset, plot_folder)

if __name__ == "__main__":
    main()
