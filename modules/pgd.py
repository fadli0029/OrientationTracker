# -------------------------------------------------------------------------
# author: muhammad fadli alim arsani
# email: fadlialim0029[at]gmail.com
# file: jax_quaternion.py
# description: this file contains the implementation of the
#              quaternion kinematics and the optimization
#              of the motion model using the projected gradient descent
#              (PGD) algorithm.
# misc: this is also part of one of the projects in the course
#       "sensing and estimation in robotics" taught by prof. nikolay
#       atanasov @uc san diego.
#       https://natanaso.github.io/ece276a/index.html
# -------------------------------------------------------------------------

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
    """
    Projected Gradient Descent (PGD) algorithm for optimizing
    the quaternion kinematics motion model.

    Args:
        q_motion:  the initial quaternion motion model being optimized
        exp_term:  the exponential term in the quaternion kinematics
        a_ts:      the accelerometer measurements
        step_size: the step size for the gradient descent
        num_iters: the number of iterations for the optimization
        eps:       the epsilon value for numerical stability

    Returns:
        q:         the optimized quaternion motion model
        cost:      the cost of the optimization
    """
    costs = []
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
    """
    Optimize the quaternion kinematics motion model using the
    IMU measurement and the projected gradient descent (PGD) algorithm.

    Args:
        dataset:               the dataset number
        processed_imu_dataset: the processed IMU dataset
        step_size:             the step size for the gradient descent
        num_iters:             the number of iterations for the optimization
        eps:                   the epsilon value for numerical stability
        debug:                 the debug flag

    Returns:
        q_optim:               the optimized quaternion motion model
        q_motion:              the initial quaternion motion model
        a_estim:               the estimated accelerometer measurements
        a_obs:                 the observed accelerometer measurements
        costs:                 the cost of the optimization
    """
    assert type(processed_imu_dataset) == dict,\
        "processed_imu_dataset must be a dictionary"

    start = time.time()

    print(f"==========> 🚀🚀🚀  Finding the optimal quaternions for dataset {dataset}")
    a_ts = processed_imu_dataset["accs"]
    w_ts = processed_imu_dataset["gyro"]
    t_ts = processed_imu_dataset["t_ts"]

    q_motion = jnp.zeros((w_ts.shape[0], 4), dtype=jnp.float64)
    q_motion = q_motion.at[:, 0].set(1.)
    q_motion, exp_term = motion_model(q_motion, w_ts, t_ts)
    a_obs = observation_model(q_motion)

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
    """
    Implement the cost function for the quaternion kinematics optimization

    Args:
        q:       the quaternion motion model
        exp:     the exponential term in the quaternion kinematics
        acc_imu: the accelerometer measurements

    Returns:
        cost:    the cost of the optimization
    """
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
    Implement the quaternion kinematics motion model

    Args:
        q: the initial quaternion motion model
        w_ts: the gyroscope measurements
        t_ts: the time stamps

    Returns:
        q: the quaternion motion model
        exp_term: the exponential term in the quaternion kinematics
    """
    tau_ts = (t_ts[1:] - t_ts[:-1]).reshape(-1, 1)
    exp_term = qexp_jax(jnp.hstack((jnp.zeros((w_ts.shape[0]-1, 1)), w_ts[:-1] * tau_ts / 2)))
    for i in range(w_ts.shape[0]-1):
        q = q.at[i+1].set(qmult_jax(q[i], exp_term[i]))
    return q, exp_term

@jit
def observation_model(qs):
    """
    Implement the observation model for the accelerometer measurements

    Args:
        qs: the quaternion motion model

    Returns:
        result: the estimated accelerometer measurements
    """
    g = jnp.array([0., 0., 0., 1.]).reshape(1, 4)
    result = qmult_jax(qinverse_jax(qs), qmult_jax(g, qs))
    return result[:, 1:]
