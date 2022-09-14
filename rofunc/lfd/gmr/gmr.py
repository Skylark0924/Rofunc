import os
import numpy as np
import matplotlib.pyplot as plt
import pbdlib as pbd
import rofunc as rf


def traj_align(demos_x, demos_dx, demos_xdx, horizon=100, plot=False):
    demos_x, demos_dx, demos_xdx = pbd.utils.align_trajectories(demos_x, [demos_dx, demos_xdx])

    t = np.linspace(0, horizon, demos_x[0].shape[0])
    if plot:
        fig, ax = plt.subplots(nrows=2)
        for d in demos_x:
            ax[0].set_prop_cycle(None)
            ax[0].plot(d)

        ax[1].plot(t)
        plt.show()
    return demos_x, demos_dx, demos_xdx, t


def GMM_learning(demos, reg=1e-3, plot=False):
    model = pbd.GMM(nb_states=4)
    model.init_hmm_kbins(demos)  # initializing model

    data = np.vstack([d for d in demos])
    model.em(data, reg=reg)

    # plotting
    if plot:
        fig, ax = plt.subplots(nrows=4)
        fig.set_size_inches(12, 7.5)

        # position plotting
        for i in range(4):
            for p in demos:
                ax[i].plot(p[:, 0], p[:, i + 1])

            pbd.plot_gmm(model.mu, model.sigma, ax=ax[i], dim=[0, i + 1])
        plt.show()
    return model


def estimate(model, demos_x, cond_input, dim_in, dim_out, plot=False):
    mu, sigma = model.condition(cond_input, dim_in=dim_in, dim_out=dim_out)

    if plot:
        pbd.plot_gmm(mu, sigma, dim=[0, 1], color='orangered', alpha=0.3)
        for d in demos_x:
            plt.plot(d[:, 0, 0], d[:, 0, 1])
        plt.show()
    return mu, sigma


def uni(demos_x, demos_dx, demos_xdx, plot=False):
    demos_x, demos_dx, demos_xdx, t = traj_align(demos_x, demos_dx, demos_xdx, plot=plot)
    demos = [np.hstack([t[:, None], d]) for d in demos_xdx]

    model = GMM_learning(demos, plot=plot)
    mu, sigma = estimate(model, demos_x, t[:, None], plot=plot)
    return model, mu, sigma


if __name__ == '__main__':
    datapath = '/home/ubuntu/Github/Knowledge-Universe/Robotics/Roadmap-for-robot-science/rofunc/lfd/src/pbd/pbdlib/data/gui/'
    data = np.load(datapath + 'test_001.npy', allow_pickle=True, encoding="latin1")[()]

    demos_x = data['x']  # Position data
    demos_dx = data['dx']  # Velocity data
    demos_xdx = [np.hstack([_x, _dx]) for _x, _dx in zip(demos_x, demos_dx)]  # Position-velocity
    uni(demos_x, demos_dx, demos_xdx, plot=True)
