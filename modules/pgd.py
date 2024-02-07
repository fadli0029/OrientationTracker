# -------------------------------------------------------------------------
# Author: Muhammad Fadli Alim Arsani
# Email: fadlialim0029[at]gmail.com
# File: pgd.py
# Description: this file contains the implementation of the
#              projected gradient descent (PGD) algorithm for traking
#              the orientation of a body.
# Misc: this is also part of one of the projects in the course
#       "sensing and estimation in robotics" taught by Prof. Nikolay
#       Atanasov @UC San Diego.
#       https://natanaso.github.io/ece276a/index.html
# -------------------------------------------------------------------------

import time
import jax
from tqdm import tqdm
import jax.numpy as jnp
from jax import value_and_grad, jit

# Assuming .jax_quaternion import includes qmult_jax, qinverse_jax, qexp_jax, qlog_jax
from .jax_quaternion import *

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

class PGD:
    def __init__(self, training_parameters):
        self.step_size = training_parameters["step_size"]
        self.num_iters = training_parameters["num_iterations"]
        self.eps = training_parameters["eps_numerical_stability"]

    def pgd(self, q_motion, exp_term, a_ts):
        costs = []
        q = q_motion[1:]
        for _ in tqdm(range(self.num_iters), desc="PGD Iterations", unit="iter"):
            cost, loss_grad = value_and_grad(self.cost_function)(q, exp_term, a_ts)
            q = q.at[:].set(q-(loss_grad*self.step_size))

            # project onto unit quaternion
            norms = jnp.sqrt(jnp.sum(q**2, axis=1, keepdims=True)) + self.eps
            q = q / norms

            costs.append(cost)
        return q, costs

    def run(self, data, q_motion, exp_term):
        start = time.time()

        a_ts, w_ts, t_ts = data
        q_optim, costs = self.pgd(
            q_motion,
            exp_term,
            a_ts
        )
        a_optim = observation_model(q_optim)

        opt_duration = time.time() - start
        print(f"Finished optimizing! Took {opt_duration:.2f} seconds.\n")
        return q_optim, a_optim

    def cost_function(self, q, exp, acc_imu):
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

# import time
# import jax
# from tqdm import tqdm
# import jax.numpy as jnp
# from .jax_quaternion import *
# from jax import value_and_grad, jit
# import transforms3d.quaternions as tq
# import transforms3d as tf3d

# class PGD:
#     def __init__(self):
#         pass

#     def pgd(self, q_motion, exp_term, a_ts, step_size=1e-3, num_iters=100, eps=1e-6):
#         costs = []
#         q = q_motion[1:]
#         for _ in tqdm(range(num_iters), desc="PGD Iterations", unit="iter"):
#             cost, loss_grad = value_and_grad(self.cost_function)(q, exp_term, a_ts)
#             q = q.at[:].set(q-(loss_grad*step_size))

#             # project onto unit quaternion
#             norms = jnp.sqrt(jnp.sum(q**2, axis=1, keepdims=True)) + eps
#             q = q / norms

#             costs.append(cost)
#         return q, costs

#     def run_pgd(self, data, q_motion, exp_term, step_size=1e-3, num_iters=100, eps=1e-6, debug=False):
#         start = time.time()

#         a_ts, w_ts, t_ts = data
#         q_optim, costs = self.pgd(
#             q_motion,
#             exp_term,
#             a_ts,
#             step_size=step_size,
#             num_iters=num_iters,
#             eps=eps
#         )
#         a_optim = self.observation_model(q_optim)

#         opt_duration = time.time() - start
#         print(f"Finished optimizing! Took {opt_duration:.2f} seconds.\n")
#         return q_optim, a_optim, costs

#     def cost_function(self, q, exp, acc_imu):
#         term_1 = 0.5 * (
#             jnp.linalg.norm(2*
#                             qlog_jax(
#                                 qmult_jax(
#                                     qinverse_jax(q[1:, :]), qmult_jax(q, exp)[:-1, :]
#                                 )
#                 )
#             )
#         )**2
#         term_2 = 0.5 * jnp.linalg.norm(acc_imu[1:, :] - self.observation_model(q))**2
#         return term_1 + term_2

#     def motion_model(self, q, w_ts, t_ts):
#         tau_ts = (t_ts[1:] - t_ts[:-1]).reshape(-1, 1)
#         exp_term = qexp_jax(jnp.hstack((jnp.zeros((w_ts.shape[0]-1, 1)), w_ts[:-1] * tau_ts / 2)))
#         for i in range(w_ts.shape[0]-1):
#             q = q.at[i+1].set(qmult_jax(q[i], exp_term[i]))
#         return q, exp_term

#     @jit
#     def observation_model(self, qs):
#         g = jnp.array([0., 0., 0., 1.]).reshape(1, 4)
#         result = qmult_jax(qinverse_jax(qs), qmult_jax(g, qs))
#         return result[:, 1:]


# import time
# import jax
# from tqdm import tqdm
# import jax.numpy as jnp
# from .jax_quaternion import *
# from jax import value_and_grad, jit
# import transforms3d.quaternions as tq
# import transforms3d as tf3d

# def pgd(
#     q_motion,
#     exp_term,
#     a_ts,
#     step_size=1e-3,
#     num_iters=100,
#     eps=1e-6
# ):
#     """
#     Projected Gradient Descent (PGD) algorithm for optimizing
#     the quaternion kinematics motion model.

#     Args:
#         q_motion:  the initial quaternion motion model being optimized
#         exp_term:  the exponential term in the quaternion kinematics
#         a_ts:      the accelerometer measurements
#         step_size: the step size for the gradient descent
#         num_iters: the number of iterations for the optimization
#         eps:       the epsilon value for numerical stability

#     Returns:
#         q:         the optimized quaternion motion model
#         cost:      the cost of the optimization
#     """
#     costs = []
#     q = q_motion[1:]
#     for _ in tqdm(range(num_iters), desc="PGD Iterations", unit="iter"):
#         cost, loss_grad = value_and_grad(cost_function)(q, exp_term, a_ts)
#         q = q.at[:].set(q-(loss_grad*step_size))

#         # project onto unit quaternion
#         norms = jnp.sqrt(jnp.sum(q**2, axis=1, keepdims=True)) + eps
#         q = q / norms

#         costs.append(cost)
#     return q, cost

# def run_pgd(
#     data,
#     q_motion,
#     exp_term,
#     step_size=1e-3,
#     num_iters=100,
#     eps=1e-6,
#     debug=False
# ):
#     """
#     Optimize the quaternion kinematics motion model using the
#     IMU measurement and the projected gradient descent (PGD) algorithm.

#     Args:
#         processed_imu_dataset: the processed IMU dataset
#         step_size:             the step size for the gradient descent
#         num_iters:             the number of iterations for the optimization
#         eps:                   the epsilon value for numerical stability
#         debug:                 the debug flag

#     Returns:
#         q_optim:               the optimized quaternion motion model
#         q_motion:              the initial quaternion motion model
#         a_estim:               the estimated accelerometer measurements
#         a_obs:                 the observed accelerometer measurements
#         costs:                 the cost of the optimization
#     """
#     start = time.time()

#     a_ts, w_ts, t_ts = data
#     q_optim, costs = pgd(
#         q_motion,
#         exp_term,
#         a_ts,
#         step_size=step_size,
#         num_iters=num_iters,
#         eps=eps
#     )
#     a_optim = observation_model(q_optim)

#     opt_duration = time.time() - start
#     print(f"Finished optimizing! Took {opt_duration:.2f} seconds.\n")
#     return q_optim, a_optim, costs

# def cost_function(q, exp, acc_imu):
#     """
#     Implement the cost function for the quaternion kinematics optimization

#     Args:
#         q:       the quaternion motion model
#         exp:     the exponential term in the quaternion kinematics
#         acc_imu: the accelerometer measurements

#     Returns:
#         cost:    the cost of the optimization
#     """
#     term_1 = 0.5 * (
#         jnp.linalg.norm(2*
#                         qlog_jax(
#                             qmult_jax(
#                                 qinverse_jax(q[1:, :]), qmult_jax(q, exp)[:-1, :]
#                             )
#             )
#         )
#     )**2
#     term_2 = 0.5 * jnp.linalg.norm(acc_imu[1:, :] - observation_model(q))**2
#     return term_1 + term_2

# def motion_model(q, w_ts, t_ts):
#     """
#     Implement the quaternion kinematics motion model

#     Args:
#         q: the initial quaternion motion model
#         w_ts: the gyroscope measurements
#         t_ts: the time stamps

#     Returns:
#         q: the quaternion motion model
#         exp_term: the exponential term in the quaternion kinematics
#     """
#     tau_ts = (t_ts[1:] - t_ts[:-1]).reshape(-1, 1)
#     exp_term = qexp_jax(jnp.hstack((jnp.zeros((w_ts.shape[0]-1, 1)), w_ts[:-1] * tau_ts / 2)))
#     for i in range(w_ts.shape[0]-1):
#         q = q.at[i+1].set(qmult_jax(q[i], exp_term[i]))
#     return q, exp_term

# @jit
# def observation_model(qs):
#     """
#     Implement the observation model for the accelerometer measurements

#     Args:
#         qs: the quaternion motion model

#     Returns:
#         result: the estimated accelerometer measurements
#     """
#     g = jnp.array([0., 0., 0., 1.]).reshape(1, 4)
#     result = qmult_jax(qinverse_jax(qs), qmult_jax(g, qs))
#     return result[:, 1:]
