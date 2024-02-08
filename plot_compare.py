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

def plot(es, dataset, plot_folder, cmd_args):

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    if not cmd_args.test:
        e_vicon = es["vicon"]

    e_pgd = es["pgd"]
    e_ekf7 = es["ekf7"]
    e_ekf4 = es["ekf4"]

    fig, ax = plt.subplots(3, 1, figsize=(20, 10))
    if not cmd_args.test:
        ax[0].plot(e_vicon[:, 0], label="Vicon", color="g")
    ax[0].plot(e_pgd[:, 0], label="PGD", color="r")
    ax[0].plot(e_ekf7[:, 0], label="7-State EKF", color="b")
    ax[0].plot(e_ekf4[:, 0], label="4-State EKF", color="y")
    ax[0].set_title(f"Roll - {dataset}")
    ax[0].legend()

    if not cmd_args.test:
        ax[1].plot(e_vicon[:, 1], label="Vicon", color="g")
    ax[1].plot(e_pgd[:, 1], label="PGD", color="r")
    ax[1].plot(e_ekf7[:, 1], label="7-State EKF", color="b")
    ax[1].plot(e_ekf4[:, 1], label="4-State EKF", color="y")
    ax[1].set_title(f"Pitch - {dataset}")
    ax[1].legend()

    if not cmd_args.test:
        ax[2].plot(e_vicon[:, 2], label="Vicon", color="g")
    ax[2].plot(e_pgd[:, 2], label="PGD", color="r")
    ax[2].plot(e_ekf7[:, 2], label="7-State EKF", color="b")
    ax[2].plot(e_ekf4[:, 2], label="4-State EKF", color="y")
    ax[2].set_title(f"Yaw - {dataset}")
    ax[2].legend()

    if cmd_args.test:
        fname = "pgd_ekf"
    else:
        fname = "pgd_ekf_vicon"
    plt.savefig(f"{plot_folder}/{fname}_{dataset}.png")

def main():
    parser = argparse.ArgumentParser(description="Plot results for comparison")
    parser.add_argument("--test", action="store_true", help="If on test mode, don't plot vicon.")
    parser.add_argument("--q_path", type=str, default="results/", help="Folder where the results are stored.")
    parser.add_argument("--plot_folder", type=str, default="plot_images/", help="Folder to save the plots.")
    parser.add_argument("--datasets", nargs="+", type=int, help="List of datasets to train and test the algorithm.")
    parser.add_argument("--path_to_datasets", type=str, default="data/trainset/", help="Path to the datasets.")

    args = parser.parse_args()
    q_path = args.q_path
    plot_folder = args.plot_folder

    # if datasets is not provided, use all datasets
    if not args.test:
        if args.datasets is None:
            datasets = [i for i in range(1, 10)]
        else:
            datasets = args.datasets
    if args.test:
        datasets = [i for i in range(10, 12)]

    data = {}
    for dataset in datasets:
        data[dataset] = {}

        fname = f"{q_path}/q_optim_{dataset}"
        data[dataset]["pgd"] = np.load(f"{fname}_PGD.npy")
        data[dataset]["ekf7"] = np.load(f"{fname}_EKF7state.npy")
        data[dataset]["ekf4"] = np.load(f"{fname}_EKF4state.npy")

    if not args.test:
        vicon_datasets = load_all_vicon_datasets(
            args.path_to_datasets, datasets
        )

    # use tqdm
    for dataset in tqdm(datasets):
        if not args.test:
            vicon = vicon_datasets[dataset]

        q_pgd = data[dataset]["pgd"]
        q_ekf7 = data[dataset]["ekf7"]
        q_ekf4 = data[dataset]["ekf4"]

        if not args.test:
            e_vicon = euler(vicon)

        e_pgd = euler(q_pgd)
        e_ekf7 = euler(q_ekf7)
        e_ekf4 = euler(q_ekf4)

        if args.test:
            es = {
                "pgd": e_pgd,
                "ekf7": e_ekf7,
                "ekf4": e_ekf4
            }
        else:
            es = {
                "vicon": e_vicon,
                "pgd": e_pgd,
                "ekf7": e_ekf7,
                "ekf4": e_ekf4
            }

        plot(es, dataset, plot_folder, args)

if __name__ == "__main__":
    main()
