from math import factorial

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pytransform3d.rotations import matrix_from_quaternion, plot_basis


def get_matrices(param, data):
    param['nbPoints'] = len(data)

    R = np.identity((param["nbData"] - 1) * param["nbVarPos"], dtype=np.float32) * param[
        "rfactor"]  # Control cost matrix

    tl = np.linspace(0, param["nbData"], param["nbPoints"] + 1)
    tl = np.rint(tl[1:]).astype(np.int64) - 1
    idx_slices = [slice(i, i + param["nb_var"], 1) for i in (tl * param["nb_var"])]

    # Target
    muQ = np.zeros((param["nb_var"] * param["nbData"], 1), dtype=np.float32)
    # Task precision
    Q = np.zeros((param["nb_var"] * param["nbData"], param["nb_var"] * param["nbData"]), dtype=np.float32)

    via_point = []
    for i in range(len(idx_slices)):
        slice_t = idx_slices[i]
        x_t = np.zeros((param["nb_var"], 1))
        x_t[:param["nbVarPos"]] = data[i].reshape((param["nbVarPos"], 1))
        muQ[slice_t] = x_t
        via_point.append(x_t)

        Q[slice_t, slice_t] = np.diag(
            np.hstack((np.ones(param["nbVarPos"]), np.zeros(param["nb_var"] - param["nbVarPos"]))))
    return via_point, muQ, Q, R, idx_slices, tl


def get_matrices_vel(param, data):
    param['nbPoints'] = len(data)

    R = np.identity((param["nbData"] - 1) * param["nbVarPos"], dtype=np.float32) * param[
        "rfactor"]  # Control cost matrix

    tl = np.linspace(0, param["nbData"], param["nbPoints"] + 1)
    tl = np.rint(tl[1:]).astype(np.int64) - 1
    idx_slices = [slice(i, i + param["nb_var"], 1) for i in (tl * param["nb_var"])]

    # Target
    muQ = np.zeros((param["nb_var"] * param["nbData"], 1), dtype=np.float32)
    # Task precision
    Q = np.zeros((param["nb_var"] * param["nbData"], param["nb_var"] * param["nbData"]), dtype=np.float32)

    via_point = []
    for i in range(len(idx_slices)):
        slice_t = idx_slices[i]
        # x_t = np.zeros((param["nb_var"], 1))
        x_t = data[i].reshape((param["nb_var"], 1))
        muQ[slice_t] = x_t
        via_point.append(x_t)

        Q[slice_t, slice_t] = np.diag(
            np.hstack((np.ones(param["nbVarPos"]), np.zeros(param["nb_var"] - param["nbVarPos"]))))
    return via_point, muQ, Q, R, idx_slices, tl


def set_dynamical_system(param):
    A1d = np.zeros((param["nbDeriv"], param["nbDeriv"]), dtype=np.float32)
    B1d = np.zeros((param["nbDeriv"], 1), dtype=np.float32)
    for i in range(param["nbDeriv"]):
        A1d += np.diag(np.ones(param["nbDeriv"] - i), i) * param["dt"] ** i * 1 / factorial(i)
        B1d[param["nbDeriv"] - i - 1] = param["dt"] ** (i + 1) * 1 / factorial(i + 1)

    A = np.kron(A1d, np.identity(param["nbVarPos"], dtype=np.float32))
    B = np.kron(B1d, np.identity(param["nbVarPos"], dtype=np.float32))

    nb_var = param["nbVarPos"] * param["nbDeriv"]  # Dimension of state vector

    # Build Sx and Su transfer matrices
    Su = np.zeros((nb_var * param["nbData"], param["nbVarPos"] * (param["nbData"] - 1)))
    Sx = np.kron(np.ones((param["nbData"], 1)), np.eye(nb_var, nb_var))

    M = B
    for i in range(1, param["nbData"]):
        Sx[i * nb_var:param["nbData"] * nb_var, :] = np.dot(Sx[i * nb_var:param["nbData"] * nb_var, :], A)
        Su[nb_var * i:nb_var * i + M.shape[0], 0:M.shape[1]] = M
        M = np.hstack((np.dot(A, M), B))  # [0,nb_state_var-1]

    return Su, Sx


def get_u_x(param, start_pose, via_point, muQ, Q, R, Su, Sx):
    x0 = start_pose.reshape((14, 1))

    # Equ. 18
    u_hat = np.linalg.inv(Su.T @ Q @ Su + R) @ Su.T @ Q @ (muQ - Sx @ x0)
    # x= S_x x_1 + S_u u
    x_hat = (Sx @ x0 + Su @ u_hat).reshape((-1, param["nb_var"]))
    return u_hat, x_hat


def plot_2d(param, x_hat_l, x_hat_r, idx_slices, tl, via_point_l, via_point_r):
    plt.figure()
    plt.title("2D Trajectory")
    plt.scatter(x_hat_l[0, 0], x_hat_l[0, 1], c='blue', s=100)
    plt.scatter(x_hat_r[0, 0], x_hat_r[0, 1], c='green', s=100)
    for slice_t in idx_slices:
        plt.scatter(param["muQ_l"][slice_t][0], param["muQ_l"][slice_t][1], c='red', s=100)
        plt.scatter(param["muQ_r"][slice_t][0], param["muQ_r"][slice_t][1], c='orange', s=100)
        plt.plot([param["muQ_l"][slice_t][0], param["muQ_r"][slice_t][0]],
                 [param["muQ_l"][slice_t][1], param["muQ_r"][slice_t][1]], linewidth=2, color='black')
    plt.plot(x_hat_l[:, 0], x_hat_l[:, 1], c='blue')
    plt.plot(x_hat_r[:, 0], x_hat_r[:, 1], c='green')
    plt.axis("off")
    plt.gca().set_aspect('equal', adjustable='box')

    fig, axs = plt.subplots(3, 1)
    for i, t in enumerate(tl):
        axs[0].scatter(t, param["muQ_l"][idx_slices[i]][0], c='red')
        axs[0].scatter(t, param["muQ_r"][idx_slices[i]][0], c='orange')
    axs[0].plot(x_hat_l[:, 0], c='blue')
    axs[0].plot(x_hat_r[:, 0], c='green')
    axs[0].set_ylabel("$x_1$")
    axs[0].set_xticks([0, param["nbData"]])
    axs[0].set_xticklabels(["0", "T"])

    for i, t in enumerate(tl):
        axs[1].scatter(t, param["muQ_l"][idx_slices[i]][1], c='red')
        axs[1].scatter(t, param["muQ_r"][idx_slices[i]][1], c='orange')
    axs[1].plot(x_hat_l[:, 1], c='blue')
    axs[1].plot(x_hat_r[:, 1], c='green')
    axs[1].set_ylabel("$x_2$")
    axs[1].set_xlabel("$t$")
    axs[1].set_xticks([0, param["nbData"]])
    axs[1].set_xticklabels(["0", "T"])

    dis_lst = []
    for i in range(len(x_hat_l)):
        dis_lst.append(np.sqrt(np.sum(np.square(x_hat_l[i, :2] - x_hat_r[i, :2]))))

    dis_lst = np.array(dis_lst)
    timestep = np.arange(len(dis_lst))
    axs[2].plot(timestep, dis_lst)
    axs[2].set_ylabel("traj_dis")
    axs[2].set_xlabel("$t$")
    axs[2].set_xticks([0, param["nbData"]])
    axs[2].set_xticklabels(["0", "T"])

    dis_lst = []
    via_point_l = np.array(via_point_l)
    via_point_r = np.array(via_point_r)
    for i in range(len(via_point_l)):
        dis_lst.append(np.sqrt(np.sum(np.square(via_point_l[i, :2] - via_point_r[i, :2]))))

    dis_lst = np.array(dis_lst)
    timestep = np.arange(len(dis_lst))
    axs[3].plot(timestep, dis_lst)

    plt.show()


def plot_3d_bi(x_hat_l, x_hat_r, muQ_l=None, muQ_r=None, idx_slices=None, ori=True, save=False, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')

    if muQ_l is not None and muQ_r is not None and idx_slices is not None:
        for slice_t in idx_slices:
            ax.scatter(muQ_l[slice_t][0], muQ_l[slice_t][1], muQ_l[slice_t][2], c='red', s=10)
            ax.scatter(muQ_r[slice_t][0], muQ_r[slice_t][1], muQ_r[slice_t][2], c='orange', s=10)

    # Plot 3d trajectories
    ax.plot(x_hat_l[:, 0], x_hat_l[:, 1], x_hat_l[:, 2], c='blue')
    ax.plot(x_hat_r[:, 0], x_hat_r[:, 1], x_hat_r[:, 2], c='green')

    # Starting points
    ax.scatter(x_hat_l[0, 0], x_hat_l[0, 1], x_hat_l[0, 2], c='blue', s=20)
    ax.scatter(x_hat_r[0, 0], x_hat_r[0, 1], x_hat_r[0, 2], c='green', s=20)

    if ori:
        l_ori = x_hat_l[:, 3:7]
        r_ori = x_hat_r[:, 3:7]
        for t in range(len(l_ori)):
            R_l = matrix_from_quaternion(l_ori[t])
            R_r = matrix_from_quaternion(r_ori[t])
            p_l = x_hat_l[t, :3]
            p_r = x_hat_r[t, :3]
            ax = plot_basis(ax=ax, R=R_l, p=p_l, s=0.01)
            ax = plot_basis(ax=ax, R=R_r, p=p_r, s=0.01)

    if save:
        np.save("/controller/data/cup.npy", np.array(x_hat_l))
        np.save("/controller/data/spoon.npy", np.array(x_hat_r))

    plt.show()


def plot_3d_uni(x_hat, muQ=None, idx_slices=None, ori=False, save=False, save_file_name=None, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d', fc='white')

    if muQ is not None and idx_slices is not None:
        # i = 0
        for slice_t in idx_slices:
            ax.scatter(muQ[slice_t][0], muQ[slice_t][1], muQ[slice_t][2], c='red', s=10)
            # label = '%d' % i
            # ax.text(muQ[slice_t][0, 0], muQ[slice_t][2, 0], muQ[slice_t][1, 0], label)
            # i += 1

    # Plot 3d trajectories
    ax.plot(x_hat[:, 0], x_hat[:, 1], x_hat[:, 2], c='black')

    # Starting points
    ax.scatter(x_hat[0, 0], x_hat[0, 1], x_hat[0, 2], c='black', s=20)

    if ori:
        ori = x_hat[:, 3:7]
        for t in range(0, len(ori), 10):
            R = matrix_from_quaternion(ori[t])
            p = x_hat[t, :3]
            ax = plot_basis(ax=ax, R=R, p=p, s=0.001)

    if save:
        assert save_file_name is not None
        np.save(save_file_name, np.array(x_hat))

    plt.show()


def uni(param, data):
    start_pose = np.zeros((14,), dtype=np.float32)
    start_pose[:7] = data[0]

    via_point_pose = data[1:]
    param['nbPoints'] = len(via_point_pose)

    via_point, muQ, Q, R, idx_slices, tl = get_matrices(param, via_point_pose)
    Su, Sx = set_dynamical_system(param)
    u_hat, x_hat = get_u_x(param, start_pose, via_point, muQ, Q, R, Su, Sx)

    return u_hat, x_hat, muQ, idx_slices


def uni_recursive(param, data, interval=3):
    start_pose = np.zeros((14,), dtype=np.float32)
    start_pose[:7] = data[0, :7]

    x_hat_lst = []
    for i in tqdm(range(0, len(data), interval)):
        via_point_pose = data[i + 1:i + interval + 1]
        param['nbPoints'] = len(via_point_pose)

        via_point, muQ, Q, R, idx_slices, tl = get_matrices_vel(param, via_point_pose)
        Su, Sx = set_dynamical_system(param)
        u_hat, x_hat = get_u_x(param, start_pose, via_point, muQ, Q, R, Su, Sx)
        start_pose = x_hat[-1]
        x_hat_lst.append(x_hat)

    x_hat = np.array(x_hat_lst).reshape((-1, 14))

    return u_hat, x_hat, muQ, idx_slices


def bi(param, l_data, r_data):
    l_start_pose = np.zeros((14,), dtype=np.float32)
    r_start_pose = np.zeros((14,), dtype=np.float32)
    l_start_pose[:7] = l_data[0]
    r_start_pose[:7] = r_data[0]
    via_point_pose_l = l_data[1:]
    via_point_pose_r = r_data[1:]
    param['nbPoints'] = len(via_point_pose_l)

    via_point_l, muQ_l, Q, R, idx_slices, tl = get_matrices(param, via_point_pose_l)
    via_point_r, muQ_r, Q, R, idx_slices, tl = get_matrices(param, via_point_pose_r)

    Su, Sx = set_dynamical_system(param)

    u_hat_l, x_hat_l = get_u_x(param, l_start_pose, via_point_l, muQ_l, Q, R, Su, Sx)
    u_hat_r, x_hat_r = get_u_x(param, r_start_pose, via_point_r, muQ_r, Q, R, Su, Sx)

    return u_hat_l, u_hat_r, x_hat_l, x_hat_r, muQ_l, muQ_r, idx_slices


if __name__ == '__main__':
    import rofunc as rf

    param = {
        "nbData": 200,  # Number of data points
        "nbVarPos": 7,  # Dimension of position data
        "nbDeriv": 2,  # Number of static and dynamic features (2 -> [x,dx])
        "dt": 1e-2,  # Time step duration
        "rfactor": 1e-8  # Control cost
    }
    param["nb_var"] = param["nbVarPos"] * param["nbDeriv"]  # Dimension of state vector

    # Uni
    # data = np.load(
    #     '/home/ubuntu/Github/DGform/interactive/skylark/stretch-31-Aug-2022-08:48:15.683806/z_manipulator_poses.npy')
    # filter_indices = [0, 1, 5, 10, 22, 36]
    # data = data[filter_indices]

    # u_hat, x_hat, muQ, idx_slices = rf.lqt.uni(param, data)
    # rf.lqt.plot_3d_uni(x_hat, muQ, idx_slices, ori=False, save=False)

    # Bi
    # data = np.loadtxt('/home/ubuntu/Github/DGform/controller/data//link7_loc_ori.txt', delimiter=', ')
    # l_data = data[0:len(data):2]
    # r_data = data[1:len(data):2]
    # u_hat_l, u_hat_r, x_hat_l, x_hat_r, muQ_l, muQ_r, idx_slices = rf.lqt.bi(param, l_data, r_data)
    # rf.lqt.plot_3d_bi(x_hat_l, x_hat_r, muQ_l, muQ_r, idx_slices, ori=False, save=False)

    # Recursive
    # data_raw = np.load('/home/ubuntu/Data/2022_09_09_Taichi/rep3_r.npy')
    # data = np.zeros((len(data_raw), 14))
    # data[:, :7] = data_raw
    # filter_indices = [i for i in range(0, len(data_raw) - 10, 5)]
    # filter_indices.append(len(data_raw) - 1)
    # data = data[filter_indices]
    # u_hat, x_hat, muQ, idx_slices = uni_recursive(param, data, interval=2)
    # rf.lqt.plot_3d_uni(x_hat, ori=False, save=True, save_file_name='/home/ubuntu/Data/2022_09_09_Taichi/lqt_rep3_r.npy')

    # Show the data
    data = np.load('/home/ubuntu/Data/2022_09_09_Taichi/lqt_rep3_l.npy')
    plot_3d_uni(data)
