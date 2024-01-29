import time
from .helpers import *
import jax.numpy as jnp

def process_imu_data(
    imu_data_raw,
    vref,
    acc_sens,
    gyro_sens,
    static_period,
    adc_max,
    g
):
    """
    Process raw IMU data to get acceleration
    and angular velocity in physical units

    Args:
        imu_data_raw: raw IMU data, shape (N, 7), last column is
                      timestamp, otherwise:
                      imu_data_raw[:, :3]  -> acc
                      imu_data_raw[:, 3:6] -> gyro

        vref: reference voltage of the IMU
        acc_sens: sensitivity of the accelerometer
        gyro_sens: sensitivity of the gyroscope
        static_period: period of time to compute the bias
        adc_max: maximum value of the ADC
        g: gravitational acceleration

    Returns:
        acc_data: acceleration in physical units
        gyro_data: angular velocity in physical units
        timestamp: timestamp of the data
    """
    # --------------------------------------------------------------
    # convert constants to appropiate units
    # --------------------------------------------------------------
    vref = vref*1000
    gyro_sens = gyro_sens*(180./jnp.pi)

    # --------------------------------------------------------------
    # compute bias
    # --------------------------------------------------------------
    # extract the static portion of the data
    static_data = imu_data_raw[
        imu_data_raw[:, -1] <= (imu_data_raw[0, -1] + static_period)
    ]

    # compute the mean of the static portion
    bias = jnp.mean(static_data[:, :-1], axis=0) # shape (6,)

    # --------------------------------------------------------------
    # compute the physical values
    # --------------------------------------------------------------
    # Compute scale factors
    acc_sf = (vref/adc_max)/acc_sens
    gyro_sf = (vref/adc_max)/gyro_sens

    # Bias correction and conversion
    acc_data = (imu_data_raw[:, :3] - bias[:3])*acc_sf
    gyro_data = (imu_data_raw[:, 3:6] - bias[3:])*gyro_sf

    acc_data *= g # convert to m/s^2
    acc_data = acc_data.at[:, 0].set(-acc_data[:, 0])
    acc_data = acc_data.at[:, 1].set(-acc_data[:, 1])

    # add gravity since we expect Az to be g during static period
    acc_data = acc_data.at[:, 2].set(acc_data[:, 2] + g)

    return acc_data, gyro_data[:, [4, 5, 3]], imu_data_raw[:, -1]

def load_all_vicon_datasets(path: str, datasets: list):
    assert len(datasets) > 0, "No datasets provided!"
    assert type(datasets) == list, "Datasets must be a list!"

    print(f"==========> Loading {len(datasets)} Vicon datasets.")
    start = time.time()
    vicon_datasets = {}
    for dataset in datasets:
        vicon_datasets[dataset] = read_dataset(dataset, path=path, data_name='vicon')
    duration = time.time() - start
    print(f"Done! Took {duration:.2f} seconds.\n")
    return vicon_datasets

def process_all_imu_datasets(
    path: str,
    datasets: list,
    vref,
    acc_sens,
    gyro_sens,
    static_period,
    adc_max,
    g
):
    assert len(datasets) > 0, "No datasets provided!"
    assert type(datasets) == list, "Datasets must be a list!"

    print(f"==========> Processing {len(datasets)} IMU datasets.")
    start = time.time()
    processed_imu_data = {}
    for dataset in datasets:
        # load the dataset
        imu_data_raw = read_dataset(dataset, path=path, data_name='imu')

        # process the raw data
        acc_data, gyro_data, timestamps = process_imu_data(
            imu_data_raw,
            vref,
            acc_sens,
            gyro_sens,
            static_period,
            adc_max,
            g
        )
        processed_imu_data[dataset] = {
            "accs": acc_data,
            "gyro": gyro_data,
            "t_ts": timestamps
        }
    duration = time.time() - start
    print(f"Done! Plotting took {duration:.2f} seconds.\n")
    return processed_imu_data
