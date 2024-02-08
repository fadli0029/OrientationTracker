# -------------------------------------------------------------------------
# Author: Muhammad Fadli Alim Arsani
# Email: fadlialim0029[at]gmail.com
# File: preprocessing.py
# Description: This is the main file to run the project. It will load the
#              config file, process the IMU datasets, find the optimal
#              quaternions for each dataset, and save the plots.
# Misc: This is also part of one of the projects in the course
#       "Sensing and Estimation in Robotics" taught by Prof. Nikolay
#       Atanasov @UC San Diego.
#       https://natanaso.github.io/ece276a/index.html
# -------------------------------------------------------------------------

import argparse
from program_driver import *

from jax import config
config.update("jax_enable_x64", True)

def main():
    """
    """
    parser = argparse.ArgumentParser(description="Orientation tracking using IMU data.")
    parser.add_argument("mode", choices=["train", "test"], help="Mode to run the algorithm.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument("--tracker", choices=["pgd", "ekf7", "ekf4"], default="pgd", help="Tracker to use.")
    parser.add_argument("--datasets", nargs="+", type=int, help="List of datasets to train and test the algorithm.")
    parser.add_argument("--plot_folder", type=str, help="Folder to save the plots.")
    parser.add_argument("--panorama_folder", type=str, help="Folder to save the panorama images.")
    parser.add_argument("--no_force_train", action="store_false", help="Force to train the algorithm even if the results are saved.")
    parser.add_argument("--use_vicon", action="store_true", help="Will use vicon data to generate panarama images if passed.")

    args = parser.parse_args()

    # Import the appropriate tracker
    if args.tracker == "ekf7":
        from modules.ekf_7states import EKF
    elif args.tracker == "ekf4":
        from modules.ekf_4states import EKF
    elif args.tracker == "pgd":
        from modules.pgd import PGD

    configs = load_config(args.config)
    configs = update_configs(configs, args)

    if args.mode == "train":
        processed_imu_datasets, vicon_datasets, camera_datasets = load_datasets(configs, mode=args.mode)
    else:
        processed_imu_datasets, _, camera_datasets = load_datasets(configs, mode=args.mode)

    q_optims, q_motion, a_optims, a_obsrvs = {}, {}, {}, {}

    # Check if there are saved results in the results folder and if user wants to force the training.
    datasets_to_train, q_optims, q_motion, a_optims, a_obsrvs = get_datasets_to_train(
        configs, q_optims, q_motion, a_optims, a_obsrvs, args.tracker
    )

    # Run orientation tracking
    if datasets_to_train:
        if args.tracker == "pgd":
            training_parameters = configs["training_parameters"]
            tracker = PGD(training_parameters)
        else:
            tracker = EKF()

        print("=====================================================")
        print(f"Performing orientation tracking using {args.tracker}")
        print("=====================================================")
        q_optims, q_motion, a_optims, a_obsrvs = run_orientation_tracking(
            tracker,
            datasets_to_train,
            processed_imu_datasets,
            q_optims,
            q_motion,
            a_optims,
            a_obsrvs
        )

    # Save all the results
    save_all_results(
        args.tracker,
        configs,
        q_optims,
        q_motion,
        a_optims,
        a_obsrvs
    )

    # Save all plots
    if args.mode == "test":
        vicon_datasets = None
    plot_all_results(
        args.tracker,
        configs,
        q_optims,
        q_motion,
        a_optims,
        a_obsrvs,
        processed_imu_datasets,
        vicon_datasets,
        configs["results"]["plot_model"]
    )

    # Build panorama images
    build_panorama_images(
        camera_datasets,
        vicon_datasets,
        processed_imu_datasets,
        q_optims,
        configs,
        args.tracker
    )

if __name__ == "__main__":
    main()
