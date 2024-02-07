from tqdm import tqdm
import numpy as np
from numpy.linalg import inv as inverse
import transforms3d.euler as tfe

class KalmanFilter:
    def __init__(self, x0, F, H, P, Q, R):
        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.P = P
        self.Q = Q
        self.R = R
        self.x = x0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.transpose() + self.Q

    def update(self, z):
        self.K = self.P @ self.H.transpose() @ inverse(self.H @ self.P @ self.H.transpose() + self.R)
        self.x = self.x + self.K @ (z - self.H @ self.x)

        I = np.eye(self.n)
        self.P = (I - self.K @ self.H) @ self.P

        return self.x

    def update_F(self, F):
        self.F = F

    def get_F(self, w, delta_t):
        gyro_phi, gyro_theta, gyro_omega = w
        F = np.array(
            np.identity(4)
            + (delta_t / 2)
            * np.array(
                [
                    [0, -gyro_phi, -gyro_theta, -gyro_omega],
                    [gyro_phi, 0, gyro_omega, -gyro_theta],
                    [gyro_theta, -gyro_omega, 0, gyro_phi],
                    [gyro_omega, gyro_theta, -gyro_phi, 0],
                ]
            )
        )
        return F

    def get_quat_from_accelerometer(self, a_x, a_y, a_z):
        g = 9.80665 # got this from scipy.constants.g
        roll = np.arcsin(a_x / g)
        pitch = -np.arcsin(a_y / (g * np.cos(roll)))
        yaw = 0.
        return tfe.euler2quat(roll, pitch, yaw)

    def run(self, data, q_mot, exp_term):
        """
        Note: for kalman filter, q_mot and exp_term are not used. It is
        only there for API consistency.
        """
        a_ts, w_ts, t_ts = data
        tau_ts = (t_ts[1:] - t_ts[:-1]).reshape(-1, 1)

        x_record = [self.x]
        # use tqdm
        for i in tqdm(range(len(tau_ts))):
            F = self.get_F(w_ts[i], tau_ts[i])
            self.update_F(F)
            self.predict()

            q_rpy = self.get_quat_from_accelerometer(*a_ts[i])

            x = self.update(q_rpy)
            x /= np.linalg.norm(x)

            x_record.append(x)

        return np.array(x_record)[1:], a_ts[1:]
