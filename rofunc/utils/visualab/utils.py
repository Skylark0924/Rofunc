import os


def set_axis(ax, labels=None, elev=45, azim=45, roll=0):
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_aspect('equal', 'box')
    if labels is None:
        labels = ['x', 'y', 'z']
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])


def save_img(fig, save_dir, fig_name=None, dpi=300, transparent=False, format=None):
    if format is None:
        format = ['eps', 'png']
    if fig_name is None:
        nb_files = len(os.listdir(save_dir))
        fig_name = 'fig_{}'.format(nb_files)
    for f in format:
        full_fig_name = '{}.{}'.format(fig_name, f)
        save_path = os.path.join(save_dir, full_fig_name)
        fig.savefig(save_path, dpi=dpi, transparent=transparent, format=f)
