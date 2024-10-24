#  Copyright (C) 2024, Junjia Liu
# 
#  This file is part of Rofunc.
# 
#  Rofunc is licensed under the GNU General Public License v3.0.
#  You may use, distribute, and modify this code under the terms of the GPL-3.0.
# 
#  Additional Terms for Commercial Use:
#  Commercial use requires sharing 50% of net profits with the copyright holder.
#  Financial reports and regular payments must be provided as agreed in writing.
#  Non-compliance results in revocation of commercial rights.
# 
#  For more details, see <https://www.gnu.org/licenses/>.
#  Contact: skylark0924@gmail.com

"""
Coordinate transformation functions with tensor support
----------------------------------------------------------
| This module provides functions to convert between different coordinate systems with tensor support.
| Note

1. Quaternions ix+jy+kz+w are represented as [x, y, z, w].
2. Euler angles are represented as [roll, pitch, yaw], in radians. The rotation order is ZYX.
3. Rotation matrices are represented as (3, 3).
4. Homogeneous matrices are represented as (4, 4).
"""

import numpy as np
import torch

# epsilon for testing whether a number is close to zero
_EPS = torch.finfo(torch.float32).eps * 4.0


def check_pos_tensor(pos):
    """
    Check if the input position is valid.

    :param pos: (batch, 3) or (3, )
    :return: position

    >>> check_pos_tensor([0, 0, 0])
    tensor([[0., 0., 0.]])
    >>> check_pos_tensor([[0, 0, 0]])
    tensor([[0., 0., 0.]])
    >>> check_pos_tensor(np.array([0, 0, 0]))
    tensor([[0., 0., 0.]])
    """
    pos = torch.tensor(pos, dtype=torch.float32)
    if len(pos.shape) == 1:
        pos = pos.unsqueeze(0)
    assert pos.shape[-1] == 3, "The last dimension of the input tensor should be 3."
    return pos


def check_quat_tensor(quat):
    """
    Check if the input quat is normalized.

    :param quat: (batch, 4) or (4, )
    :return: normalized quat

    >>> check_quat_tensor([0, 5, 0, 1])
    tensor([[0.0000, 0.9806, 0.0000, 0.1961]])
    >>> check_quat_tensor([[0, 2, 0, 1]])
    tensor([[0.0000, 0.8944, 0.0000, 0.4472]])
    >>> check_quat_tensor(np.array([1, 5, 5.435, 1]))
    tensor([[0.1330, 0.6650, 0.7228, 0.1330]])
    """
    quat = torch.tensor(quat, dtype=torch.float32)
    if len(quat.shape) == 1:
        quat = quat.unsqueeze(0)
    assert quat.shape[-1] == 4, "The last dimension of the input tensor should be 4."
    norm = torch.norm(quat, dim=-1, keepdim=True)
    if torch.any(norm == 0):
        raise ValueError(
            f"The input quat is invalid. The index of the invalid quat is {torch.where(norm == 0)[0]}")
    quat = quat / (norm + _EPS)
    return quat


def check_rot_matrix_tensor(rot_matrix):
    """
    Check if the input rotation matrix is valid, orthogonal, and normalize it if necessary.

    :param rot_matrix: Input rotation matrix
    :return: Validated and normalized rotation matrix

    >>> from rofunc.utils.robolab.coord.transform import random_rot_matrix
    >>> rot_matrix = random_rot_matrix() * 3
    >>> torch.allclose(check_rot_matrix_tensor(rot_matrix) * 3, torch.tensor(rot_matrix, dtype=torch.float32))
    True
    """
    rot_matrix = torch.tensor(rot_matrix, dtype=torch.float32)
    if len(rot_matrix.shape) == 2:
        rot_matrix = rot_matrix.unsqueeze(0)
    # Check if the matrix is square
    if rot_matrix.shape[-1] != rot_matrix.shape[-2]:
        raise ValueError("Input matrix is not square.")

    # # Check orthogonality: R^T * R should be equal to the identity matrix
    # identity_matrix = torch.eye(rot_matrix.shape[-1])
    # matrix_product = torch.matmul(rot_matrix.transpose(-1, -2), rot_matrix)
    # if not torch.allclose(matrix_product, identity_matrix):
    #     raise ValueError("Input matrix is not orthogonal.")

    # Normalize the matrix if necessary
    normalized_rot_matrix = rot_matrix
    column_norms = torch.norm(normalized_rot_matrix, dim=-2)
    if not torch.allclose(column_norms, torch.ones_like(column_norms)):
        normalized_rot_matrix = normalized_rot_matrix / column_norms.unsqueeze(-1)

    return normalized_rot_matrix


def check_euler_tensor(euler):
    """
    Check if the input euler angles are valid.

    :param euler: (batch, 3) or (3, )
    :return: euler angles

    >>> check_euler_tensor([1.57, 0, 0])
    tensor([[1.5700, 0.0000, 0.0000]])
    >>> check_euler_tensor([[0, 0, 0]])
    tensor([[0., 0., 0.]])
    >>> check_euler_tensor(np.array([0, 0, 0]))
    tensor([[0., 0., 0.]])
    """
    euler = torch.tensor(euler, dtype=torch.float32)
    if len(euler.shape) == 1:
        euler = euler.unsqueeze(0)
    assert euler.shape[-1] == 3, "The last dimension of the input tensor should be 3."
    return euler


def random_quat_tensor(batch_size, rand=None):
    """
    Return uniform random unit quat.

    :param batch_size: Batch size
    :param rand: Random number generator (optional)
    :return: Random unit quat, [x, y, z, w]

    >>> torch.allclose(torch.norm(random_quat_tensor(100), dim=-1), torch.ones(100))
    True
    >>> rand_quat = random_quat_tensor(100)
    >>> torch.allclose(check_quat_tensor(rand_quat), rand_quat)
    True
    """
    if rand is None:
        rand = torch.rand

    random_values = rand(batch_size, 3)
    r1 = torch.sqrt(1 - random_values[:, 0])
    r2 = torch.sqrt(random_values[:, 0])
    pi2 = 2 * torch.pi
    t1 = pi2 * random_values[:, 1]
    t2 = pi2 * random_values[:, 2]

    x = r1 * torch.sin(t1)
    y = r1 * torch.cos(t1)
    z = r2 * torch.sin(t2)
    w = r2 * torch.cos(t2)

    quat = torch.stack([x, y, z, w], dim=-1)

    return quat


def random_rot_matrix_tensor(batch_size, rand=None):
    """
    Generate random rotation matrix. quat = [x, y, z, w].

    :param batch_size: Batch size
    :param rand: Random number generator (optional)
    :return: Random rotation matrix

    >>> rand_rot_matrix = random_rot_matrix_tensor(100)
    >>> torch.allclose(rand_rot_matrix.det(), torch.ones(100))
    True
    >>> torch.allclose(check_rot_matrix_tensor(rand_rot_matrix), rand_rot_matrix)
    True
    >>> from rofunc.utils.robolab.coord.transform import check_rot_matrix
    >>> torch.allclose(torch.tensor(check_rot_matrix(rand_rot_matrix[0]), dtype=torch.float32), rand_rot_matrix[0])
    True
    """
    if rand is None:
        rand = torch.rand

    quat = random_quat_tensor(batch_size, rand)

    wx, wy, wz, ww = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    xx = wx * wx
    xy = wx * wy
    xz = wx * wz
    xw = wx * ww
    yy = wy * wy
    yz = wy * wz
    yw = wy * ww
    zz = wz * wz
    zw = wz * ww

    rot_matrix = torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - zw),
            2 * (xz + yw),
            2 * (xy + zw),
            1 - 2 * (xx + zz),
            2 * (yz - xw),
            2 * (xz - yw),
            2 * (yz + xw),
            1 - 2 * (xx + yy),
        ],
        dim=1,
    ).view(batch_size, 3, 3)

    return rot_matrix


def quat_from_rot_matrix_tensor(rot_matrix):
    """
    Convert rotation matrix to quat. [x, y, z, w]

    :param rot_matrix:
    :return: quat, [x, y, z, w]

    >>> quat_from_rot_matrix_tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    tensor([[0., 0., 0., 1.]])
    >>> quat_from_rot_matrix_tensor([[0.9362934, -0.2896295,  0.1986693], [0.3129918,  0.9447025, -0.0978434], [-0.1593451,  0.1537920,  0.9751703]])
    tensor([[0.0641, 0.0912, 0.1534, 0.9819]])
    >>> rand_rot_matrix = random_rot_matrix_tensor(100)
    >>> torch.allclose(check_rot_matrix_tensor(rand_rot_matrix), rot_matrix_from_quat_tensor(quat_from_rot_matrix_tensor(rand_rot_matrix)), rtol=1e-03, atol=1e-03)
    True
    """
    rot_matrix = check_rot_matrix_tensor(rot_matrix)

    trace = rot_matrix[:, 0, 0] + rot_matrix[:, 1, 1] + rot_matrix[:, 2, 2]
    r = torch.sqrt(1 + trace)

    qw = 0.5 * r
    qx = (rot_matrix[:, 2, 1] - rot_matrix[:, 1, 2]) / (2 * r)
    qy = (rot_matrix[:, 0, 2] - rot_matrix[:, 2, 0]) / (2 * r)
    qz = (rot_matrix[:, 1, 0] - rot_matrix[:, 0, 1]) / (2 * r)

    quat = torch.stack([qx, qy, qz, qw], dim=-1)
    return quat


def quat_from_euler_tensor(euler):
    """
    Convert euler angles to quat. The rotation order is ZYX.

    :param euler: (batch, 3) or (3, ), [roll, pitch, yaw], the rotation order is ZYX.
    :return: quat, [x, y, z, w]

    >>> quat_from_euler_tensor([0, 0, 0])
    tensor([[0., 0., 0., 1.]])
    >>> quat_from_euler_tensor([[0, 0, 0]])
    tensor([[0., 0., 0., 1.]])
    >>> quat_from_euler_tensor(np.array([0, 0, 0]))
    tensor([[0., 0., 0., 1.]])
    >>> quat_from_euler_tensor([[0, 1.23, 0.57], [0.5, 0.3, 0.7], [0.1, 0.2, 0.3]])
    tensor([[-0.1622,  0.5537,  0.2296,  0.7838],
            [ 0.1801,  0.2199,  0.2938,  0.9126],
            [ 0.0343,  0.1060,  0.1436,  0.9833]])
    """
    euler = check_euler_tensor(euler)

    roll = euler[:, 0]
    pitch = euler[:, 1]
    yaw = euler[:, 2]

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    quat = torch.stack([qx, qy, qz, qw], dim=-1)

    return quat


def rot_matrix_from_quat_tensor(quat):
    """
    Convert quat to rotation matrix.

    :param quat: [x, y, z, w]
    :return:

    >>> rot_matrix_from_quat_tensor([0, 0, 0, 1])
    tensor([[[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]]])
    >>> rot_matrix_from_quat_tensor([[0.06146124, 0, 0, 0.99810947], [0.2794439, 0.0521324, 0.3632374, 0.8872722]])
    tensor([[[ 1.0000,  0.0000,  0.0000],
             [ 0.0000,  0.9924, -0.1227],
             [ 0.0000,  0.1227,  0.9924]],
    <BLANKLINE>
            [[ 0.7307, -0.6154,  0.2955],
             [ 0.6737,  0.5799, -0.4580],
             [ 0.1105,  0.5338,  0.8384]]])
    """
    quat = check_quat_tensor(quat)
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    r11 = 1 - 2 * (y ** 2 + z ** 2)
    r12 = 2 * (x * y - z * w)
    r13 = 2 * (x * z + y * w)
    r21 = 2 * (x * y + z * w)
    r22 = 1 - 2 * (x ** 2 + z ** 2)
    r23 = 2 * (y * z - x * w)
    r31 = 2 * (x * z - y * w)
    r32 = 2 * (y * z + x * w)
    r33 = 1 - 2 * (x ** 2 + y ** 2)

    rot_matrix = torch.stack([
        torch.stack([r11, r12, r13], dim=-1),
        torch.stack([r21, r22, r23], dim=-1),
        torch.stack([r31, r32, r33], dim=-1)
    ], dim=-2)

    return check_rot_matrix_tensor(rot_matrix)


def rot_matrix_from_euler_tensor(euler):
    """
    Convert euler angles to rotation matrix.

    :param euler: (batch, 3) or (3, ), [roll, pitch, yaw] in radian
    :return: Rotation matrix

    >>> rot_matrix_from_euler_tensor([0, 0, 0])
    tensor([[[1., 0., 0.],
             [0., 1., 0.],
             [-0., 0., 1.]]])
    >>> rot_matrix_from_euler_tensor([[0.5, 0.3, 0.7], [1.33, 0.2, -0.03]])
    tensor([[[ 0.7307, -0.4570,  0.5072],
             [ 0.6154,  0.7625, -0.1996],
             [-0.2955,  0.4580,  0.8384]],
    <BLANKLINE>
            [[ 0.9796,  0.2000,  0.0182],
             [-0.0294,  0.2326, -0.9721],
             [-0.1987,  0.9518,  0.2337]]])
    """
    euler = check_euler_tensor(euler)

    roll = euler[:, 0]
    pitch = euler[:, 1]
    yaw = euler[:, 2]

    cos_r = torch.cos(roll)
    sin_r = torch.sin(roll)
    cos_p = torch.cos(pitch)
    sin_p = torch.sin(pitch)
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)

    batch_size = euler.size(0)
    device = euler.device

    rot_matrix = torch.zeros((batch_size, 3, 3), device=device)
    rot_matrix[:, 0, 0] = cos_y * cos_p
    rot_matrix[:, 0, 1] = cos_y * sin_p * sin_r - sin_y * cos_r
    rot_matrix[:, 0, 2] = cos_y * sin_p * cos_r + sin_y * sin_r
    rot_matrix[:, 1, 0] = sin_y * cos_p
    rot_matrix[:, 1, 1] = sin_y * sin_p * sin_r + cos_y * cos_r
    rot_matrix[:, 1, 2] = sin_y * sin_p * cos_r - cos_y * sin_r
    rot_matrix[:, 2, 0] = -sin_p
    rot_matrix[:, 2, 1] = cos_p * sin_r
    rot_matrix[:, 2, 2] = cos_p * cos_r

    return rot_matrix


def euler_from_quat_tensor(quat):
    """
    Convert quat to euler angles.

    :param quat: [x, y, z, w]
    :return: euler angles, [roll, pitch, yaw] in radian

    >>> euler_from_quat_tensor(torch.tensor([[0, 0, 0, 1.]]))
    tensor([[0., 0., 0.]])
    >>> euler_from_quat_tensor(torch.tensor([[0.06146124, 0, 0, 0.99810947], [0.2794439, 0.0521324, 0.3632374, 0.8872722]]))
    tensor([[ 0.1230,  0.0000,  0.0000],
            [ 0.5669, -0.1107,  0.7449]])
    """
    quat = check_quat_tensor(quat)

    qx = quat[:, 0]
    qy = quat[:, 1]
    qz = quat[:, 2]
    qw = quat[:, 3]

    roll = torch.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = torch.asin(torch.clamp(2 * (qw * qy - qz * qx), -1, 1))
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    euler = torch.stack([roll, pitch, yaw], dim=-1)

    return euler


def euler_from_rot_matrix_tensor(rot_matrix):
    """
    Convert rotation matrix to euler angles.

    :param rot_matrix:
    :return:

    >>> euler_from_rot_matrix_tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    tensor([[0., -0., 0.]])
    >>> euler_from_rot_matrix_tensor([[[ 0.7307, -0.4570,  0.5072], [ 0.6154,  0.7625, -0.1996], [-0.2955,  0.4580,  0.8384]], [[ 0.9796,  0.2000,  0.0182], [-0.0294,  0.2326, -0.9721], [-0.1987,  0.9518,  0.2337]]])
    tensor([[ 0.5000,  0.3000,  0.6999],
            [ 1.3300,  0.2000, -0.0300]])
    """
    rot_matrix = check_rot_matrix_tensor(rot_matrix)

    r11, r12, r13 = rot_matrix[:, 0, 0], rot_matrix[:, 0, 1], rot_matrix[:, 0, 2]
    r21, r22, r23 = rot_matrix[:, 1, 0], rot_matrix[:, 1, 1], rot_matrix[:, 1, 2]
    r31, r32, r33 = rot_matrix[:, 2, 0], rot_matrix[:, 2, 1], rot_matrix[:, 2, 2]

    pitch = -torch.asin(r31)
    roll = torch.atan2(r32 / torch.cos(pitch), r33 / torch.cos(pitch))
    yaw = torch.atan2(r21 / torch.cos(pitch), r11 / torch.cos(pitch))

    euler = torch.stack([roll, pitch, yaw], dim=-1)

    return euler


def homo_matrix_from_quat_tensor(quat, pos=None):
    """
    Convert quat and pos to homogeneous matrix

    :param quat:
    :param pos:
    :return:
    """
    quat = check_quat_tensor(quat)
    if pos is not None:
        pos = check_pos_tensor(pos)
        assert quat.shape[0] == pos.shape[0]
    else:
        pos = torch.zeros((quat.shape[0], 3))

    batch_size = quat.shape[0]
    device = quat.device

    homo_matrix = torch.zeros((batch_size, 4, 4), device=device)
    rot_matrix = rot_matrix_from_quat_tensor(quat)
    homo_matrix[:, :3, :3] = rot_matrix
    homo_matrix[:, :3, 3] = pos
    homo_matrix[:, 3, 3] = 1
    return homo_matrix
