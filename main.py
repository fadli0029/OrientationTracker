import yaml
from modules.pgd import optimize
from modules.preprocessing import *
from modules.helpers import save_plot

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
        data_processing_constants["adc_max"],
        other_configs["gravity_constant"]
    )

    # load all vicon datasets
    vicon_datasets = load_all_vicon_datasets(
        other_configs["path_to_datasets"],
        other_configs["datasets"]
    )

    # some bookeeping for plotting. We will track the following
    # for all datasets:
    # - costs:    the cost over iterations during optimization
    # - q_optims: the optimized quaternion
    # - q_motion: the quaternions estimate from motion model
    # - a_estims: the acceleration from motion model
    # - a_obsrvs: the acceleration estimate from observation model
    q_optims, q_motion, a_estims, a_obsrvs, costs_record = {}, {}, {}, {}, {}
    for dataset in other_configs["datasets"]:
        q_opt, q_mot, a_est, a_obs, costs = optimize(
            dataset,
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

    # Save the plots
    print("==========> 📊  Saving plots")
    for dataset in other_configs["datasets"]:
        save_plot(
            q_optims[dataset],
            q_motion[dataset],
            a_estims[dataset],
            a_obsrvs[dataset],
            vicon_datasets[dataset],
            processed_imu_datasets[dataset]["accs"],
            dataset
        )
    print("🎉🎉🎉  Done!")

if __name__ == "__main__":
    main()
