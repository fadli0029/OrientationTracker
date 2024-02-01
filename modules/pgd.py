import time
import jax
import jax.numpy as jnp
from .jax_quaternion import *
from jax import value_and_grad, jit
import transforms3d.quaternions as tq
import transforms3d as tf3d

def pgd(
    q_motion,
    exp_term,
    a_ts,
    step_size=1e-3,
    num_iters=100,
    eps=1e-6
):
    costs = [] # track cost over iterations
    q = q_motion[1:]
    for _ in range(num_iters):
        cost, loss_grad = value_and_grad(cost_function)(q, exp_term, a_ts)
        q = q.at[:].set(q-(loss_grad*step_size))

        # project onto unit quaternion
        norms = jnp.sqrt(jnp.sum(q**2, axis=1, keepdims=True)) + eps
        q = q / norms

        costs.append(cost)
    return q, cost

def optimize(
    dataset: int,
    processed_imu_dataset: dict,
    step_size=1e-3,
    num_iters=100,
    eps=1e-6,
    debug=False
):
    assert type(processed_imu_dataset) == dict,\
        "processed_imu_dataset must be a dictionary"

    start = time.time()

    print(f"==========> 🚀🚀🚀  Optimizing dataset {dataset}")
    a_ts = processed_imu_dataset["accs"]
    w_ts = processed_imu_dataset["gyro"]
    t_ts = processed_imu_dataset["t_ts"]

    q_motion = jnp.zeros((w_ts.shape[0], 4), dtype=jnp.float64)
    q_motion = q_motion.at[:, 0].set(1.)
    q_motion, exp_term = motion_model(q_motion, w_ts, t_ts)
    a_obs = observation_model(q_motion)

    # Print all shapes
    q_optim, costs = pgd(
        q_motion,
        exp_term,
        a_ts,
        step_size=step_size,
        num_iters=num_iters,
        eps=eps
    )
    a_estim = observation_model(q_optim)

    opt_duration = time.time() - start
    print(f"Finished optimizing! Took {opt_duration:.2f}.")
    print("")
    return q_optim, q_motion, a_estim, a_obs, costs

def cost_function(q, exp, acc_imu):
    term_1 = 0.5 * (
        jnp.linalg.norm(2*
                        qlog_jax(
                            qmult_jax(
                                qinverse_jax(q[1:, :]), qmult_jax(q, exp)[:-1, :]
                            )
            )
        )
    )**2
    term_2 = 0.5 * jnp.linalg.norm(acc_imu[1:, :] - observation_model(q))**2
    return term_1 + term_2

def motion_model(q, w_ts, t_ts):
    """
    Implements quaternion kinematics motion model
    given angular velocities w_ts and the differences between
    consecutive time stamps tau_ts from calibrated imu data.
    """
    tau_ts = (t_ts[1:] - t_ts[:-1]).reshape(-1, 1)
    exp_term = qexp_jax(jnp.hstack((jnp.zeros((w_ts.shape[0]-1, 1)), w_ts[:-1] * tau_ts / 2)))
    for i in range(w_ts.shape[0]-1):
        q = q.at[i+1].set(qmult_jax(q[i], exp_term[i]))
    return q, exp_term

@jit
def observation_model(qs):
    """
    Observation model for quaternion-based motion.

    Parameters:
    qs (array): Quaternion array.

    Returns:
    array: Observed acceleration.
    """
    g = jnp.array([0., 0., 0., 1.]).reshape(1, 4)
    result = qmult_jax(qinverse_jax(qs), qmult_jax(g, qs))
    return result[:, 1:]

def sensor_fusion(gyro_data, accel_data, delta_t, alpha=0.5):
    """
    Implements drift correction via complementary filtering on gyroscopic and accelerometer data for
    roll, pitch, and yaw angles.

    Parameters:
    gyro_data (array): Gyroscopic data with shape (N, 3).
    accel_data (array): Accelerometer data with shape (N, 3).
    delta_t (array): Time deltas with shape (N-1,).

    Returns:
    array: Fused sensor data for roll, pitch, and yaw angles.
    """
    N = gyro_data.shape[0]
    angles = jnp.zeros((N, 3))  # Initialize roll, pitch, yaw angles array

    # Calculate initial roll and pitch from the accelerometer data
    roll = jnp.arctan2(accel_data[0, 1], accel_data[0, 2])
    pitch = jnp.arctan2(-accel_data[0, 0], jnp.sqrt(accel_data[0, 1]**2 + accel_data[0, 2]**2))
    yaw = 0.0  # Initialize yaw to 0 as we cannot estimate it from accelerometer

    angles = angles.at[0].set(jnp.array([roll, pitch, yaw]))

    for i in range(1, N):
        # Gyro integration for each axis
        gyro_angles = angles[i-1] + gyro_data[i-1] * delta_t[i-1]

        # Accelerometer angle calculations
        roll = jnp.arctan2(accel_data[i, 1], accel_data[i, 2])
        pitch = jnp.arctan2(-accel_data[i, 0], jnp.sqrt(accel_data[i, 1]**2 + accel_data[i, 2]**2))
        # Yaw remains integrated gyro value as there's no accelerometer data for it

        # Apply complementary filter for roll and pitch, and gyro integration for yaw
        fused_roll = alpha * gyro_angles[0] + (1 - alpha) * roll
        fused_pitch = alpha * gyro_angles[1] + (1 - alpha) * pitch
        fused_yaw = gyro_angles[2]  # Assuming we have no other sensor data for yaw

        angles = angles.at[i].set(jnp.array([fused_roll, fused_pitch, fused_yaw]))

    return jnp.array([tf3d.euler.euler2quat(*row) for row in angles])
