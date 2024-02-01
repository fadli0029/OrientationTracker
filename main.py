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
from modules.utils import save_plot

from jax import config
config.update("jax_enable_x64", True)

def load_config(path_to_config):
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)
    return config

def main(path_to_config="config.yaml"):
    # load config
    config = load_config(path_to_config)
    data_processing_constants = config["data_processing_constants"]
    training_parameters = config["training_parameters"]
    other_configs = config["other_configs"]

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

    # Find the optimal quaternions for each dataset
    print("===============================")
    print("Performing orientation tracking")
    print("===============================")
    start = time.time()
    q_optims, q_motion, a_estims, a_obsrvs, costs_record = {}, {}, {}, {}, {}
    for dataset in other_configs["datasets"]:
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
    # Find duration in minutes and seconds
    minutes = int(duration // 60)
    seconds = duration % 60
    print(f"==========> ✅  Done! Total duration for {len(other_configs['datasets'])} datasets: {minutes}m {seconds:.2f}s")
    print("")

    # Save the plots
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
            other_configs["save_image_folder"]
        )

        iter_end = time.time()
        iter_duration = iter_end - iter_start

        # Update progress bar with iteration duration
        pbar.set_postfix(time=f"{iter_duration:.4f}s")
    print(f"==========> ✅  Done! All plots saved to {other_configs['save_image_folder']}")

if __name__ == "__main__":
    main()
