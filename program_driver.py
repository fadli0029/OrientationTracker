import os
import time
import yaml
from tqdm import tqdm
from modules.pgd import motion_model, observation_model
from modules.utils import *
from modules.panorama import *
from modules.preprocessing import *

def load_config(path_to_config):
    with open(path_to_config, "r") as f:
        configs = yaml.safe_load(f)
    return configs

def update_configs(configs, args):
    configs["other_configs"]["no_force_train"] = args.no_force_train
    configs["other_configs"]["use_vicon"] = args.use_vicon
    if args.plot_folder:
        configs["results"]["plot_folder"] = args.plot_folder
    if args.panorama_folder:
        configs["results"]["panorama_folder"] = args.panorama_folder

    if args.mode == "test":
        configs["other_configs"]["path_to_datasets"] = "data/testset/"
        configs["other_configs"]["datasets"] = [10, 11]
    elif args.mode == "train":
        configs["other_configs"]["path_to_datasets"] = "data/trainset/"
        if args.datasets:
            configs["other_configs"]["datasets"] = args.datasets

    return configs

def load_datasets(configs, mode):
    print("==================================")
    print("Loading datasets...")
    print("==================================")

    data_processing_constants = configs["data_processing_constants"]
    other_configs = configs["other_configs"]

    # load and process all imu datasets
    processed_imu_datasets = process_all_imu_datasets(
        other_configs["path_to_datasets"],
        other_configs["datasets"],
        data_processing_constants["vref"],
        data_processing_constants["acc_sensitivity"],
        data_processing_constants["gyro_sensitivity"],
        data_processing_constants["static_period"],
        data_processing_constants["adc_max"]
    )

    if mode == "train":
        # load all vicon datasets
        vicon_datasets = load_all_vicon_datasets(
            other_configs["path_to_datasets"],
            other_configs["datasets"]
        )
    else:
        vicon_datasets = None

    # load all camera datasets
    camera_datasets = load_all_camera_datasets(
        other_configs["path_to_datasets"],
        other_configs["datasets"]
    )

    return processed_imu_datasets, vicon_datasets, camera_datasets

def build_panorama_images(
    camera_datasets,
    vicon_datasets,
    processed_imu_datasets,
    q_optims,
    configs
):
    other_configs = configs["other_configs"]

    print("==================================")
    print("Building panorama images...")
    print("==================================")
    panorama_image_record = {}
    start = time.time()
    for dataset in list(camera_datasets.keys()):
        if other_configs["use_vicon"]:
            R = np.concatenate((np.identity(3)[np.newaxis, :, :], vicon_datasets[dataset]["rots"]), axis=0)
            ts = vicon_datasets[dataset]["ts"]
            prefix = "vicon"
        else:
            R = np.array(quat2rot(np.vstack((np.array([1., 0., 0., 0.]), q_optims[dataset]))))
            ts = processed_imu_datasets[dataset]["t_ts"]
            prefix = "quaternion"
        panorama_img = build_panorama(
            camera_datasets[dataset],
            R,
            ts,
            dataset,
            prefix,
            other_configs["panorama_images_folder"]
        )
        panorama_image_record[dataset] = panorama_img
    end = time.time()
    duration = round(end - start, 2)
    print(f"==========> ✅  Done (Took {duration}s)! All panorama images saved to {other_configs['panorama_images_folder']}")


def run_orientation_tracking(
    tracker,
    datasets_to_train,
    processed_imu_datasets,
    q_optims,
    q_motion,
    a_optims,
    a_obsrvs
):
    start = time.time()
    for dataset in datasets_to_train:
        print(f"==========> 🚀  Finding the optimal quaternions for dataset {dataset}")

        # Unpack the processed imu data
        a_ts = processed_imu_datasets[dataset]["accs"]
        w_ts = processed_imu_datasets[dataset]["gyro"]
        t_ts = processed_imu_datasets[dataset]["t_ts"]

        # Initialize the quaternions and initial quaternions and acceleration
        # from model dynamics
        q_mot = jnp.zeros((w_ts.shape[0], 4), dtype=jnp.float64)
        q_mot = q_mot.at[:, 0].set(1.)
        q_mot, exp_term = motion_model(q_mot, w_ts, t_ts)
        a_obs = observation_model(q_mot)

        # Save the initial quaternions and initial acceleration estimate
        q_motion[dataset] = q_mot
        a_obsrvs[dataset] = a_obs

        data = (a_ts, w_ts, t_ts)

        # -----------------------------------------------------------------------------
        q_optim, a_optim = tracker.run(
            data,
            q_mot,
            exp_term
        )
        # -----------------------------------------------------------------------------
        q_optims[dataset]     = q_optim
        a_optims[dataset]     = a_optim
    end = time.time()
    duration = end - start
    minutes = int(duration // 60)
    seconds = duration % 60
    print(f"==========> ✅  Done! Total duration for {len(datasets_to_train)} datasets: {minutes}m {seconds:.2f}s\n")

    return q_optims, q_motion, a_optims, a_obsrvs

def save_all_results(
    tracker,
    configs,
    q_optims,
    q_motion,
    a_optims,
    a_obsrvs
):
    print("==================================")
    print("Saving results...")
    print("==================================")
    results = {
        configs["results"]["optimized_quaternion_fname"]:    q_optims,
        configs["results"]["motion_model_quaternion_fname"]: q_motion,
        configs["results"]["acceleration_estimate_fname"]:   a_optims,
        configs["results"]["obsrv_model_fname"]:             a_obsrvs
    }
    pbar = tqdm(results.items(), desc="==========> 📝  Saving results", unit="data")
    for f, data in pbar:
        save_results(data, f, configs["results"]["folder"], tracker)
    print(f"==========> ✅  Done! All results saved to {configs['results']['folder']}\n")

def plot_all_results(
    tracker,
    configs,
    q_optims,
    q_motion,
    a_optims,
    a_obsrvs,
    processed_imu_datasets,
    vicon_datasets,
    plot_model=False
):
    print("==================================")
    print("Saving plots...")
    print("==================================")
    other_configs = configs["other_configs"]
    pbar = tqdm(other_configs["datasets"], desc="==========> 📊  Saving plots", unit="plot")
    for dataset in pbar:
        iter_start = time.time()

        if vicon_datasets is not None:
            save_plot(
                tracker,
                q_optims[dataset],
                q_motion[dataset],
                a_optims[dataset],
                a_obsrvs[dataset],
                processed_imu_datasets[dataset]["accs"],
                dataset,
                other_configs["plot_figures_folder"],
                vicon_datasets[dataset],
                plot_model=plot_model
            )
        else:
            save_plot(
                tracker,
                q_optims[dataset],
                q_motion[dataset],
                a_optims[dataset],
                a_obsrvs[dataset],
                processed_imu_datasets[dataset]["accs"],
                dataset,
                other_configs["plot_figures_folder"],
                plot_model=plot_model
            )

        iter_end = time.time()
        iter_duration = iter_end - iter_start

        pbar.set_postfix(time=f"{iter_duration:.4f}s")
    print(f"==========> ✅  Done! All plots saved to {other_configs['plot_figures_folder']}\n")


def get_datasets_to_train(
    configs,
    q_optims,
    q_motion,
    a_optims,
    a_obsrvs
):
    other_configs = configs["other_configs"]
    results_configs = configs["results"]

    datasets_to_train = [] # datasets that need to be trained
    if os.path.exists(results_configs["folder"]):
        if not other_configs["no_force_train"]:
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
                results_configs["obsrv_model_fname"]
            ]

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
                                a_optims[dataset] = load_results(fname)
                            elif i == 3:
                                a_obsrvs[dataset] = load_results(fname)
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
    return datasets_to_train, q_optims, q_motion, a_optims, a_obsrvs
