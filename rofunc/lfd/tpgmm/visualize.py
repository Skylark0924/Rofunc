import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pbdlib as pbd


def hmm_plot(demos_xdx_f, model):
    fig, ax = plt.subplots(ncols=2, nrows=1)
    fig.set_size_inches(12, 6)

    # position plotting
    ax[0].set_title('pos - coord. %d' % 1)
    for p in demos_xdx_f:
        ax[0].plot(p[:, 0, 0], p[:, 0, 1])
    pbd.plot_gmm(model.mu, model.sigma, ax=ax[0], dim=[0, 1], color='steelblue')

    ax[1].set_title('pos - coord. %d' % 2)
    for p in demos_xdx_f:
        ax[1].plot(p[:, 1, 0], p[:, 1, 1])
    pbd.plot_gmm(model.mu, model.sigma, ax=ax[1], dim=[4, 5], color='orangered')

    plt.tight_layout()
    plt.show()


def hmm_plot_3d(demos_xdx_f, model):
    fig = plt.figure(figsize=(4, 4))
    fig.set_size_inches(12, 6)

    # position plotting
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title('pos - coord. %d' % 1)
    for p in demos_xdx_f:
        ax.plot(p[:, 0, 0], p[:, 0, 1], p[:, 0, 2])
    # pbd.plot_gmm(model.mu, model.sigma, ax=ax[0], dim=[0, 1], color='steelblue')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title('pos - coord. %d' % 2)
    for p in demos_xdx_f:
        ax.plot(p[:, 1, 0], p[:, 1, 1], p[:, 1, 2])
    # pbd.plot_gmm(model.mu, model.sigma, ax=ax[1], dim=[4, 5], color='orangered')

    plt.tight_layout()
    plt.show()


def poe_plot(model, mod1, mod2, prod, demos_x, demo_idx):
    fig, ax = plt.subplots(ncols=3, nrows=1)
    fig.set_size_inches((12, 6))
    for i in ax:
        i.set_aspect('equal')

    ax[0].set_title('model 1')
    pbd.plot_gmm(model.mu, model.sigma, swap=True, ax=ax[0], dim=[0, 1], color='steelblue', alpha=0.3)
    ax[1].set_title('model 2')
    ax[2].set_title('tranformed models and product')
    pbd.plot_gmm(mod1.mu, mod1.sigma, swap=True, ax=ax[2], dim=[0, 1], color='steelblue', alpha=0.3)
    pbd.plot_gmm(mod2.mu, mod2.sigma, swap=True, ax=ax[2], dim=[0, 1], color='orangered', alpha=0.3)
    pbd.plot_gmm(prod.mu, prod.sigma, swap=True, ax=ax[2], dim=[0, 1], color='gold', alpha=0.3)
    ax[2].plot(demos_x[demo_idx][:, 0], demos_x[demo_idx][:, 1], color="b")

    patches = [mpatches.Patch(color='steelblue', label='transformed model 1'),
               mpatches.Patch(color='orangered', label='transformed model 2'),
               mpatches.Patch(color='gold', label='product')]

    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def poe_plot_3d(model, mod1, mod2, prod, demos_x, demo_idx):
    fig, ax = plt.subplots(ncols=3, nrows=1)
    fig.set_size_inches((12, 6))
    for i in ax:
        i.set_aspect('equal')

    ax[0].set_title('model 1')
    pbd.plot_gmm(model.mu, model.sigma, swap=True, ax=ax[0], dim=[0, 1], color='steelblue', alpha=0.3)
    ax[1].set_title('model 2')
    ax[2].set_title('tranformed models and product')
    pbd.plot_gmm(mod1.mu, mod1.sigma, swap=True, ax=ax[2], dim=[0, 1], color='steelblue', alpha=0.3)
    pbd.plot_gmm(mod2.mu, mod2.sigma, swap=True, ax=ax[2], dim=[0, 1], color='orangered', alpha=0.3)
    pbd.plot_gmm(prod.mu, prod.sigma, swap=True, ax=ax[2], dim=[0, 1], color='gold', alpha=0.3)
    ax[2].plot(demos_x[demo_idx][:, 0], demos_x[demo_idx][:, 1], color="b")

    patches = [mpatches.Patch(color='steelblue', label='transformed model 1'),
               mpatches.Patch(color='orangered', label='transformed model 2'),
               mpatches.Patch(color='gold', label='product')]

    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def generate_plot(xi, prod, demos_x, demo_idx):
    plt.figure()

    plt.title('Trajectory reproduction')
    plt.plot(xi[:, 0], xi[:, 1], color='r', lw=2, label='generated line')
    pbd.plot_gmm(prod.mu, prod.sigma, swap=True, dim=[0, 1], color='gold')
    plt.plot(demos_x[demo_idx][:, 0], demos_x[demo_idx][:, 1], 'k--', lw=2, label='demo line')
    plt.axis('equal')
    plt.legend()
    plt.show()


def generate_plot_3d(xi, prod, demos_x, demo_idx):
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title('Trajectory reproduction')
    ax.plot(xi[:, 0], xi[:, 1], xi[:, 2], color='r', lw=2, label='generated line')
    # pbd.plot_gmm(prod.mu, prod.sigma, swap=True, dim=[0, 1], color='gold')
    ax.plot(demos_x[demo_idx][:, 0], demos_x[demo_idx][:, 1], demos_x[demo_idx][:, 2], 'k--', lw=2, label='demo line')
    plt.legend()
    plt.show()

    import numpy as np
    t = np.arange(len(xi))
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(t, xi[:, 3], color='r', lw=2, label='generated line')
    plt.plot(t, demos_x[demo_idx][:, 3], 'k--', lw=2, label='demo line')
    plt.title('w-t')

    plt.subplot(2, 2, 2)
    plt.plot(t, xi[:, 4], color='r', lw=2, label='generated line')
    plt.plot(t, demos_x[demo_idx][:, 4], 'k--', lw=2, label='demo line')
    plt.title('x-t')

    plt.subplot(2, 2, 3)
    plt.plot(t, xi[:, 5], color='r', lw=2, label='generated line')
    plt.plot(t, demos_x[demo_idx][:, 5], 'k--', lw=2, label='demo line')
    plt.title('y-t')

    plt.subplot(2, 2, 4)
    plt.plot(t, xi[:, 6], color='r', lw=2, label='generated line')
    plt.plot(t, demos_x[demo_idx][:, 6], 'k--', lw=2, label='demo line')
    plt.title('z-t')
    plt.show()
