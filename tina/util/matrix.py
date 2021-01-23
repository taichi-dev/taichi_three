from ..common import *


def identity():
    return np.eye(4)


def affine(lin, pos):
    lin = np.concatenate([lin, np.zeros((1, 3))], axis=0)
    pos = np.concatenate([pos, np.ones(1)])
    lin = np.concatenate([lin, pos[:, None]], axis=1)
    return lin


def lookat(pos=(0, 0, 0), back=(0, 0, 3), up=(0, 1, 1e-12)):
    pos = np.array(pos, dtype=float)
    back = np.array(back, dtype=float)
    up = np.array(up, dtype=float)

    fwd = -back
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    lin = np.transpose(np.stack([right, up, -fwd]))
    return np.linalg.inv(affine(lin, (pos + back)))


def ortho(left=-1, right=1, bottom=-1, top=1, near=-100, far=100):
    lin = np.eye(4)
    lin[0, 0] = 2 / (right - left)
    lin[1, 1] = 2 / (top - bottom)
    lin[2, 2] = -2 / (far - near)
    lin[0, 3] = -(right + left) / (right - left)
    lin[1, 3] = -(top + bottom) / (top - bottom)
    lin[2, 3] = -(far + near) / (far - near)
    return lin


def frustum(left=-1, right=1, bottom=-1, top=1, near=1, far=100):
    lin = np.eye(4)
    lin[0, 0] = 2 * near / (right - left)
    lin[1, 1] = 2 * near / (top - bottom)
    lin[0, 2] = (right + left) / (right - left)
    lin[1, 2] = (top + bottom) / (top - bottom)
    lin[2, 2] = -(far + near) / (far - near)
    lin[2, 3] = -2 * far * near / (far - near)
    lin[3, 2] = -1
    lin[3, 3] = 0
    return lin


def orthogonal(size=1, aspect=1, near=-100, far=100):
    ax, ay = size * aspect, size
    return ortho(-ax, ax, -ay, ay, near, far)


def perspective(fov=60, aspect=1, near=0.05, far=500):
    fov = np.tan(np.radians(fov) / 2)
    ax, ay = fov * aspect, fov
    return frustum(-near * ax, near * ax, -near * ay, near * ay, near, far)


def scale(factor):
    return affine(np.eye(3) * np.array(factor), np.zeros(3))


def translate(offset):
    return affine(np.eye(3), np.array(offset) * np.ones(3))


# https://zhuanlan.zhihu.com/p/259999988
def quaternion(q):
    R = np.array([
        [1.0 - 2 * (q[1] * q[1] + q[2] * q[2]),
            2 * (q[0] * q[1] - q[3] * q[2]),
            2 * (q[3] * q[1] + q[0] * q[2])],
        [2 * (q[0] * q[1] + q[3] * q[2]),
            1.0 - 2 * (q[0] * q[0] + q[2] * q[2]),
            2 * (q[1] * q[2] - q[3] * q[0])],
        [2 * (q[0] * q[2] - q[3] * q[1]),
            2 * (q[1] * q[2] + q[3] * q[0]),
            1.0 - 2 * (q[0] * q[0] + q[1] * q[1])]])
    return affine(R, np.zeros(3))


# https://zhuanlan.zhihu.com/p/259999988
def eularXYZ(theta):
    R_x = np.array([[1, 0, 0],
        [0, np.cos(theta[0]), -np.sin(theta[0])],
        [0, np.sin(theta[0]), np.cos(theta[0])]])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
        [0, 1, 0], [-np.sin(theta[1]), 0, np.cos(theta[1])]])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
        [np.sin(theta[2]), np.cos(theta[2]), 0], [0, 0, 1]])
    return affine(R_z @ R_y @ R_x, np.zeros(3))
