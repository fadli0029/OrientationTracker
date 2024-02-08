import numpy as np
from transforms3d.quaternions import qmult, quat2mat, qinverse
from tqdm import tqdm

class EKF:
    """
    Implementation of EKF where the state
    is of 4 elements: [q0, q1, q2, q3]
    where q0, q1, q2, q3 are the quaternion elements.
    """
    def __init__(
        self,
        q_noise = 1.0e-2,
        gyro_noise=100,
        acc_noise=0.1,
    ):
        """
        Note for gyro_noise:

        2.25 is also good. The higher the better it handles discontinuities,
        but at the cost of noisy estimates for data that has not much discontinuities.

        i.e: it's a tradeoff between handling discontinuities and noise. if you care
        about having good estimates at places where there are sudden jumps, then
        make it high. If you care about smooth estimates at places where there are
        no sudden jumps, then make it low.
        """
        self.x = np.array([1., 0., 0., 0.])

        self.P = np.identity(4) * q_noise
        self.Q = np.identity(3) * gyro_noise
        self.R = np.identity(3) * acc_noise

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
        self.P = F @ self.P @ F.T + W @ self.Q @ W.T

    def update(self, z_t):
        """
        Update step of the EKF.

        z_t: (3,) - Accelerometer measurements
        """
        # normalize z
        z_t = z_t / np.linalg.norm(z_t)

        H = self.H_mat(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ (z_t - self.h_func(self.x))
        self.x = self.x / np.linalg.norm(self.x)

        self.P = (np.eye(4) - K @ H) @ self.P

    def f_func(self, x_t, w_t, tau_t):
        """
        Process model.

        x_t: (7,)
        w_t: (3,)
        tau_t: float
        """
        q = x_t
        q = q / np.linalg.norm(q)

        dw_t = w_t * tau_t
        theta = np.linalg.norm(dw_t)
        u = dw_t/np.linalg.norm(dw_t)

        # define taylor expansion of sin and cos
        # to avoid division by zero
        if theta < 1e-6:
            qw_t = np.array([
                1,
                0.5*dw_t[0],
                0.5*dw_t[1],
                0.5*dw_t[2]
            ])
        else:
            qw_t = np.array([
                np.cos(theta/2),
                u[0]*np.sin(theta/2),
                u[1]*np.sin(theta/2),
                u[2]*np.sin(theta/2)
            ])

        qdot = qmult(q, qw_t)
        qdot = qdot / np.linalg.norm(qdot)

        return qdot

    def h_func(self, x):
        """
        Measurement model.

        x_t: (7,)
        """
        q = x
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
        wx, wy, wz = w_t

        Omega = np.array([
            [0, -wx, -wy, -wz],
            [wx, 0, wz, -wy],
            [wy, -wz, 0, wx],
            [wz, wy, -wx, 0]
        ]) * 0.5

        F = np.identity(4) + (Omega * tau_t)
        return F

    def H_mat(self, x):
        """
        Jacobian of the measurement model.

        x: (7,)
        """
        qw, qx, qy, qz = x

        H = np.zeros((3, 4))
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
        qw, qx, qy, qz = x_t

        W = np.zeros((4, 3))

        W[0, 0] = -qx*tau_t
        W[0, 1] = -qy*tau_t
        W[0, 2] = -qz*tau_t

        W[1, 0] = qw*tau_t
        W[1, 1] = -qz*tau_t
        W[1, 2] = qy*tau_t

        W[2, 0] = qz*tau_t
        W[2, 1] = qw*tau_t
        W[2, 2] = -qx*tau_t

        W[3, 0] = -qy*tau_t
        W[3, 1] = qx*tau_t
        W[3, 2] = qw*tau_t

        W = W * 0.5
        return W
