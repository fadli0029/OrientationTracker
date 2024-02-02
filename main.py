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

import yaml
import time
from tqdm import tqdm
from modules.pgd import optimize
from modules.preprocessing import *
from modules.panorama import build_panorama
from modules.utils import save_plot, load_results

from jax import config
config.update("jax_enable_x64", True)

def load_config(path_to_config):
    """
    Load the configuration file.

    Args:
        path_to_config (str): path to the configuration file.

    Returns:
        config (dict): the configuration file.
    """
    with open(path_to_config, "r") as f:
        configs = yaml.safe_load(f)
    return configs

def main(path_to_config="config.yaml"):
    """
    The main function to run the orientation tracking
    algorithm.

    Args:
        path_to_config (str): path to the configuration file.

    Returns:
        None
    """
    # load config
    config = load_config(path_to_config)
    data_processing_constants = config["data_processing_constants"]
    training_parameters = config["training_parameters"]
    other_configs = config["other_configs"]
    results_configs = config["results"]

    print("==================================")
    print("Loading datasets...")
    print("==================================")
    # load and process all IMU datasets
    processed_imu_datasets = process_all_imu_datasets(
        other_configs["path_to_datasets"],
        other_configs["datasets"],
        data_processing_constants["vref"],
        data_processing_constants["acc_sensitivity"],
        data_processing_constants["gyro_sensitivity"],
        data_processing_constants["static_period"],
        data_processing_constants["adc_max"]
    )

    # load all vicon datasets
    vicon_datasets = load_all_vicon_datasets(
        other_configs["path_to_datasets"],
        other_configs["datasets"]
    )

    # load all camera datasets
    camera_datasets = load_all_camera_datasets(
        other_configs["path_to_datasets"],
        other_configs["datasets"]
    )

    q_optims, q_motion, a_estims, a_obsrvs, costs_record = {}, {}, {}, {}, {}

    # Check if there are saved results in the results folder.
    # If there are, don't do the optimization again.
    if os.path.exists(results_configs["folder"]):
        if not other_configs["force_train"]:
            print("==================================")
            print("Loading saved results...")
            print("==================================")
            start = time.time()

            saved_results = check_files_exist(other_configs["datasets"], results_configs["folder"])
            for dataset in other_configs["datasets"]:
                if saved_results[dataset]:
                    print(f"Dataset {dataset} has saved results, will be loaded.")
                else:
                    print(f"Dataset {dataset} doesn't have saved results, will be trained.")

            results_fnames = [
                results_configs["optimized_quaternion_fname"],
                results_configs["motion_model_quaternion_fname"],
                results_configs["acceleration_estimate_fname"],
                results_configs["obsrv_model_fname"],
                results_configs["costs_record_fname"]
            ]

            datasets_to_train = [] # datasets that need to be trained
            for dataset in tqdm(other_configs["datasets"], desc="Loading saved results..."):
                if saved_results[dataset]:
                    for i, f in enumerate(results_fnames):
                        fname = results_configs["folder"] + f + "_" + str(dataset) + ".npy"
                        if os.path.exists(fname):
                            if i == 0:
                                q_optims[dataset] = load_results(fname)
                            elif i == 1:
                                q_motion[dataset] = load_results(fname)
                            elif i == 2:
                                a_estims[dataset] = load_results(fname)
                            elif i == 3:
                                a_obsrvs[dataset] = load_results(fname)
                            elif i == 4:
                                costs_record[dataset] = load_results(fname)
                else:
                    datasets_to_train.append(dataset)
            end = time.time()
            duration = end - start
            minutes = int(duration // 60)
            seconds = duration % 60
            print(f"==========> ✅  Done! Took {minutes}m {seconds:.2f}s to load the results for {len(other_configs['datasets'])} datasets\n")
        else:
            datasets_to_train = other_configs["datasets"]
    else:
        datasets_to_train = other_configs["datasets"]

    if datasets_to_train:
        print("==================================")
        print("Performing orientation tracking...")
        print("==================================")
        start = time.time()
        for dataset in datasets_to_train:
            print(f"==========> 🚀  Finding the optimal quaternions for dataset {dataset}")
            q_opt, q_mot, a_est, a_obs, costs = optimize(
                processed_imu_datasets[dataset],
                step_size=training_parameters["step_size"],
                num_iters=training_parameters["num_iterations"],
                eps=training_parameters["eps_numerical_stability"]
            )
            q_optims[dataset]     = q_opt
            q_motion[dataset]     = q_mot
            a_estims[dataset]     = a_est
            a_obsrvs[dataset]     = a_obs
            costs_record[dataset] = costs
        end = time.time()
        duration = end - start
        minutes = int(duration // 60)
        seconds = duration % 60
        print(f"==========> ✅  Done! Total duration for {len(other_configs['datasets'])} datasets: {minutes}m {seconds:.2f}s\n")

        print("==================================")
        print("Saving results...")
        print("==================================")
        results = {
            results_configs["optimized_quaternion_fname"]: q_optims,
            results_configs["motion_model_quaternion_fname"]: q_motion,
            results_configs["acceleration_estimate_fname"]: a_estims,
            results_configs["obsrv_model_fname"]: a_obsrvs,
            results_configs["costs_record_fname"]: costs_record
        }
        pbar = tqdm(results.items(), desc="==========> 📝  Saving results", unit="data")
        for f, data in pbar:
            save_results(data, f, results_configs["folder"])
        print(f"==========> ✅  Done! All results saved to {results_configs['folder']}\n")

    print("==================================")
    print("Saving plots...")
    print("==================================")
    pbar = tqdm(other_configs["datasets"], desc="==========> 📊  Saving plots", unit="plot")
    for dataset in pbar:
        iter_start = time.time()

        save_plot(
            q_optims[dataset],
            q_motion[dataset],
            a_estims[dataset],
            a_obsrvs[dataset],
            vicon_datasets[dataset],
            processed_imu_datasets[dataset]["accs"],
            dataset,
            other_configs["plot_figures_folder"]
        )

        iter_end = time.time()
        iter_duration = iter_end - iter_start

        pbar.set_postfix(time=f"{iter_duration:.4f}s")
    print(f"==========> ✅  Done! All plots saved to {other_configs['plot_figures_folder']}\n")

    # print("==================================")
    # print("Building panorama images...")
    # print("==================================")
    # panorama_image_record = {}
    # start = time.time()
    # for dataset in list(camera_datasets.keys()):
    #     panorama_img = build_panorama(
    #         camera_datasets[dataset],
    #         q_optims[dataset],
    #         processed_imu_datasets[dataset]["t_ts"],
    #         dataset,
    #         other_configs["panorama_images_folder"]
    #     )
    #     panorama_image_record[dataset] = panorama_img
    # end = time.time()
    # duration = round(end - start, 2)
    # print(f"==========> ✅  Done (Took {duration}s)! All panorama images saved to {other_configs['panorama_images_folder']}")

if __name__ == "__main__":
    main()
