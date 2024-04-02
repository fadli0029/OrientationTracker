"""
PyTeapot module for drawing rotating cube using OpenGL as per
quaternion or yaw, pitch, roll angles received over serial port.

source - modified from the original code by:
    https://github.com/thecountoftuscany/PyTeapot-Quaternion-Euler-cube-rotation
"""

import sys
import math
import pickle
import pygame
import argparse
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

from transforms3d.quaternions import mat2quat

# set true for using quaternions, false for using y,p,r angles
useQuat = True

def read_data(filename: str):
    d = []
    with open(filename, 'rb') as f:
        if sys.version_info[0] < 3:
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='latin1') # needed for python 3
    return d

def rot_to_quat(data):
    return np.array([mat2quat(data[i]) for i in range(data.shape[0])])

def load_data(filename: str, vicon=False, ignore_time_steps=800):
    if not vicon:
        data = np.load(filename)
        data = data[ignore_time_steps:-ignore_time_steps]
        return data
    else:
        vicon_data = read_data(filename)
        vicon_data = vicon_data['rots']
        vicon_data = np.transpose(vicon_data, (2, 0, 1))
        vicon_data = rot_to_quat(vicon_data)
        vicon_data = vicon_data[ignore_time_steps:-ignore_time_steps]
        return vicon_data

def dataset_range(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 9:
        raise argparse.ArgumentTypeError(f"{value} is an invalid dataset number. Please choose a value between 1 and 9.")
    return ivalue

def main():
    parser = argparse.ArgumentParser(description='PyTeapot IMU orientation visualization')
    parser.add_argument("--tracker", choices=["pgd", "ekf7", "ekf4", "vicon"], default="pgd", help="Tracker to use.")
    parser.add_argument("--dataset", type=dataset_range, help="Dataset to visualize. Please choose a value between 1 and 9.")
    parser.add_argument("--loop", action="store_true", help="Loop the visualization.")

    video_flags = OPENGL | DOUBLEBUF
    pygame.init()
    screen = pygame.display.set_mode((640, 480), video_flags)
    pygame.display.set_caption("PyTeapot IMU orientation visualization")
    resizewin(640, 480)
    init()
    frames = 0
    ticks = pygame.time.get_ticks()

    args = parser.parse_args()
    tracker = args.tracker.upper()
    if tracker != "PGD" and tracker != "VICON":
        tracker += "state"
    dataset = args.dataset
    loop = args.loop

    if tracker == "VICON":
        data_fname = f"data/trainset/vicon/viconRot{dataset}.p"
        data = load_data(data_fname, vicon=True)
    else:
        data_fname = f"results/q_optim_{dataset}_{tracker}.npy"
        data = load_data(data_fname)
    while 1:
        event = pygame.event.poll()
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            break
        if(useQuat) and frames < data.shape[0]:
            [w, nx, ny, nz] = data[frames]
        else:
            if frames < data.shape[0]:
                [yaw, pitch, roll] = data[frames]
        if(useQuat):
            draw(w, nx, ny, nz, frames, tracker, dataset)
        else:
            draw(1, yaw, pitch, roll, frames, tracker, dataset)

        # if finished reading data, then read data again (i.e: loop)
        if frames == data.shape[0] and loop:
            if tracker == "VICON":
                data = load_data(data_fname, vicon=True)
            else:
                data = load_data(data_fname)
            frames = 0
            continue
        elif frames == data.shape[0]:
            break

        pygame.display.flip()
        frames += 1
    print("fps: %d" % ((frames*1000)/(pygame.time.get_ticks()-ticks)))

def resizewin(width, height):
    """
    For resizing window
    """
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0*width/height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def init():
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

def draw(w, nx, ny, nz, frames, tracker, dataset):
    tracker_map = {
        "PGD": "Projected Gradient Descent",
        "EKF7state": "7-state Extended Kalman Filter",
        "EKF4state": "4-state Extended Kalman Filter",
        "VICON": "VICON"
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0, 0.0, -7.0)

    drawText((-2.6, 1.8, 2), "Tracker: %s" %tracker_map[tracker], 18)
    drawText((-2.6, 1.6, 2), "Dataset: %d" %dataset, 16)
    drawText((-2.6, -2, 2), "Press Escape to exit.", 16)

    if(useQuat):
        [yaw, pitch , roll] = quat_to_ypr([w, nx, ny, nz])
        drawText((-2.6, -1.8, 2), "Yaw: %.2f, Pitch: %.2f, Roll: %.2f" %(yaw, pitch, roll), 16)
        drawText((-2.6, -1.6, 2), "Time step: %d" %frames, 16)
        glRotatef(2 * math.acos(w) * 180.00/math.pi, -1 * nx, nz, ny)
    else:
        yaw = nx
        pitch = ny
        roll = nz
        drawText((-2.6, -1.8, 2), "Yaw: %.2f, Pitch: %.2f, Roll: %.2f" %(yaw, pitch, roll), 16)
        drawText((-2.6, -1.6, 2), "Time step: %d" %frames, 16)
        glRotatef(-roll, 0.00, 0.00, 1.00)
        glRotatef(pitch, 1.00, 0.00, 0.00)
        glRotatef(yaw, 0.00, 1.00, 0.00)

    glBegin(GL_QUADS)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(1.0, 0.2, -1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(1.0, 0.2, 1.0)

    glColor3f(1.0, 0.5, 0.0)
    glVertex3f(1.0, -0.2, 1.0)
    glVertex3f(-1.0, -0.2, 1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(1.0, -0.2, -1.0)

    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(1.0, 0.2, 1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(-1.0, -0.2, 1.0)
    glVertex3f(1.0, -0.2, 1.0)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(1.0, -0.2, -1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(1.0, 0.2, -1.0)

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(-1.0, -0.2, 1.0)

    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(1.0, 0.2, -1.0)
    glVertex3f(1.0, 0.2, 1.0)
    glVertex3f(1.0, -0.2, 1.0)
    glVertex3f(1.0, -0.2, -1.0)
    glEnd()


def drawText(position, textString, size):
    font = pygame.font.SysFont("Courier", size, True)
    textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

def quat_to_ypr(q):
    yaw   = math.atan2(2.0 * (q[1] * q[2] + q[0] * q[3]), q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3])
    pitch = -math.asin(2.0 * (q[1] * q[3] - q[0] * q[2]))
    roll  = math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3])
    pitch *= 180.0 / math.pi
    yaw   *= 180.0 / math.pi
    yaw   -= -0.13  # Declination at Chandrapur, Maharashtra is - 0 degress 13 min
    roll  *= 180.0 / math.pi
    return [yaw, pitch, roll]

if __name__ == '__main__':
    main()
