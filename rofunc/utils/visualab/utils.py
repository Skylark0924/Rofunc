def set_axis(ax, labels=None):
    ax.view_init(elev=45, azim=45, roll=0)
    ax.set_aspect('equal', 'box')
    if labels is None:
        labels = ['x', 'y', 'z']
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
