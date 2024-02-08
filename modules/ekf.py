import numpy as np
from transforms3d.quaternions import qmult, quat2mat, qinverse
from tqdm import tqdm

class EKF:
    """
    Implementation of EKF where the state
    is of 7 elements: [q0, q1, q2, q3, bx, by, bz]
    where q0, q1, q2, q3 are the quaternion elements
    and bx, by, bz are the gyro biases.
    """
    def __init__(
        self,
        q_noise_init=1.0e-6,
        gyro_noise_init=1.0e-6,
        gyro_noise=[3.732e-8, 3.732e-8, 3.732e-8],
        gyro_bias_drift=[1.0e-9, 1.0e-9, 1.0e-9],
        acc_noise=[1.125e-5, 1.125e-5, 4.5e-5]
    ):
        """
        The noises are taken from datasheet.
        """
        self.x = np.array([1., 0., 0., 0., 0., 0., 0.])

        gyro_noise=[0.0006109**2, 0.0006109**2, 0.0006109**2]
        gyro_bias_drift=[3.0e-7, 3.0e-7, 5.0e-7]
        acc_noise=[1.0, 1.0, 1.0]
        d = 3.0 # debugging
        acc_noise = [d * x for x in acc_noise]

        self.R = np.array([
            [acc_noise[0], 0., 0.],
            [0., acc_noise[1], 0.],
            [0., 0., acc_noise[2]]
        ])

        self.P = np.zeros((7, 7))
        self.P[:4, :4] = np.eye(4) * q_noise_init
        self.P[4:7, 4:7] = np.eye(3) * gyro_noise_init

        self.Q = np.array([
            [gyro_noise[0], 0., 0.],
            [0., gyro_noise[1], 0.],
            [0., 0., gyro_noise[2]]
        ])

        self.Q_bias = np.zeros((7, 7))
        self.Q_bias[4:7, 4:7] = np.array([
            [gyro_bias_drift[0], 0., 0.],
            [0., gyro_bias_drift[1], 0.],
            [0., 0., gyro_bias_drift[2]]
        ])

    def run(self, data, q_mot, exp_term):
        """
        Note: for kalman filter, q_mot and exp_term are not used. It is
        only there for API consistency.
        """
        a_ts, w_ts, t_ts = data
        tau_ts = (t_ts[1:] - t_ts[:-1]).reshape(-1, 1)

        x_record = [self.x]
        for i in tqdm(range(len(tau_ts))):
            self.predict(w_ts[i], tau_ts[i])
            self.update(a_ts[i])
            x_record.append(self.x)

        # return only the quaternion elements
        return np.array(x_record)[1:, :4], a_ts[1:]

    def predict(self, w_t, tau_t):
        """
        Predict step of the EKF.

        w_t: (3,) - Gyro measurements
        tau_t: float - Time step
        """
        F = self.F_mat(self.x, w_t, tau_t)
        W = self.W_mat(self.x, w_t, tau_t)

        self.x = self.f_func(self.x, w_t, tau_t)
        self.P = F @ self.P @ F.T + W @ self.Q @ W.T + self.Q_bias

    def update(self, z_t):
        """
        Update step of the EKF.

        z_t: (3,) - Accelerometer measurements
        """
        H = self.H_mat(self.x)
        V = self.V_mat(self.x)
        S = H @ self.P @ H.T + V @ self.R @ V.T
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ (z_t - self.h_func(self.x))
        self.x[:4] = self.x[:4] / np.linalg.norm(self.x[:4])

        self.P = (np.eye(7) - K @ H) @ self.P


    def f_func(self, x_t, w_t, tau_t):
        """
        Process model.

        x_t: (7,)
        w_t: (3,)
        tau_t: float
        """
        q, b = x_t[:4], x_t[4:]
        q = q / np.linalg.norm(q)

        dw_t = ((w_t - b) * tau_t)*0.5
        qw_t = np.array([1, dw_t[0], dw_t[1], dw_t[2]])
        qdot = qmult(q, qw_t)
        qdot = qdot / np.linalg.norm(qdot)

        return np.concatenate([qdot, b])

    def h_func(self, x):
        """
        Measurement model.

        x_t: (7,)
        """
        q = x[:4]
        g = np.array([0., 0., 0., 1.]).reshape(4,)
        q_inv = qinverse(q)
        res = qmult(q_inv, qmult(g, q))
        return res[1:]

    def F_mat(self, x_t, w_t, tau_t):
        """
        Jacobian of the process model.

        x_t: (7,)
        w_t: (3,)
        tau_t: float
        """
        qw, qx, qy, qz = x_t[:4]
        bx, by, bz = x_t[4:]
        wx, wy, wz = w_t

        F = np.zeros((7, 7))
        np.fill_diagonal(F, 1)

        F[0, 1] = (0.5*tau_t) * (-wx + bx)
        F[0, 2] = (0.5*tau_t) * (-wy + by)
        F[0, 3] = (0.5*tau_t) * (-wz + bz)
        F[0, 4] = 0.5*tau_t*qx
        F[0, 5] = 0.5*tau_t*qy
        F[0, 6] = 0.5*tau_t*qz

        F[1, 0] = (0.5*tau_t) * (wx - bx)
        F[1, 2] = (0.5*tau_t) * (wz - bz)
        F[1, 3] = (0.5*tau_t) * (-wy + by)
        F[1, 4] = -0.5*tau_t*qw
        F[1, 5] = 0.5*tau_t*qz
        F[1, 6] = -0.5*tau_t*qy

        F[2, 0] = (0.5*tau_t) * (wy - by)
        F[2, 1] = (0.5*tau_t) * (-wz + bz)
        F[2, 3] = (0.5*tau_t) * (wx - bx)
        F[2, 4] = -0.5*tau_t*qz
        F[2, 5] = -0.5*tau_t*qw
        F[2, 6] = 0.5*tau_t*qx

        F[3, 0] = (0.5*tau_t) * (wz - bz)
        F[3, 1] = (0.5*tau_t) * (wy - by)
        F[3, 2] = (0.5*tau_t) * (-wx + bx)
        F[3, 4] = 0.5*tau_t*qy
        F[3, 5] = -0.5*tau_t*qx
        F[3, 6] = -0.5*tau_t*qw

        return F

    def H_mat(self, x):
        """
        Jacobian of the measurement model.

        x: (7,)
        """
        qw, qx, qy, qz = x[:4]

        H = np.zeros((3, 7))
        H[0, 0] = -2*qy
        H[0, 1] = 2*qz
        H[0, 2] = -2*qw
        H[0, 3] = 2*qx

        H[1, 0] = 2*qx
        H[1, 1] = 2*qw
        H[1, 2] = 2*qz
        H[1, 3] = 2*qy

        H[2, 0] = 2*qw
        H[2, 1] = -2*qx
        H[2, 2] = -2*qy
        H[2, 3] = 2*qz

        return H

    def W_mat(self, x_t, w_t, tau_t):
        """
        Jacobian of the process model w.r.t. the noise.

        x_t: (7,)
        w_t: (3,)
        tau_t: float
        """
        qw, qx, qy, qz = x_t[:4]
        bx, by, bz = x_t[4:]
        wx, wy, wz = w_t

        W = np.zeros((7, 3))
        W[0, 0] = -0.5*tau_t*qx
        W[0, 1] = -0.5*tau_t*qy
        W[0, 2] = -0.5*tau_t*qz

        W[1, 0] = 0.5*tau_t*qw
        W[1, 1] = -0.5*tau_t*qz
        W[1, 2] = 0.5*tau_t*qy

        W[2, 0] = 0.5*tau_t*qz
        W[2, 1] = 0.5*tau_t*qw
        W[2, 2] = -0.5*tau_t*qx

        W[3, 0] = -0.5*tau_t*qy
        W[3, 1] = 0.5*tau_t*qx
        W[3, 2] = 0.5*tau_t*qw

        return W

    def V_mat(self, x_t):
        """
        Jacobian of the process model w.r.t. the noise.

        x_t: (7,)
        w_t: (3,)
        tau_t: float
        """
        return np.eye(3)
