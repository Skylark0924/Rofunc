import numpy as np

def plot_skeleton(ax, global_positions, parent_indices):
    for i in range(len(global_positions)):
        if parent_indices[i] != -1:
            parent_pos = global_positions[parent_indices[i]]
            joint_pos = global_positions[i]
            ax.plot([parent_pos[0], joint_pos[0]], [parent_pos[1], joint_pos[1]], [parent_pos[2], joint_pos[2]], 'k-')

def plot_ellipsoid(ax, eigenvalues, eigenvectors, center, color):
    eigenvalues = eigenvalues[-3:]
    eigenvectors = eigenvectors[-3:, -3:]
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v)) / 5
    y = np.outer(np.sin(u), np.sin(v)) / 5
    z = np.outer(np.ones_like(u), np.cos(v)) / 5
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = np.dot(eigenvectors, np.array([x[i, j], y[i, j], z[i, j]]) * np.sqrt(eigenvalues))
            x[i, j] += center[0]
            y[i, j] += center[1]
            z[i, j] += center[2]
    ax.plot_surface(x, y, z, color=color, alpha=0.3)

def ua_get_color(score):
    if 1 <= score <= 2:
        return 'green'
    elif 3 <= score <= 4:
        return 'orange'
    elif 5 <= score <= 6:
        return 'red'
    else:
        return 'purple'  # Other scores in purple

def la_get_color(score):
    if 1 <= score < 2:
        return 'green'
    elif 2 <= score:
        return 'orange'
    else:
        return 'purple'  # Other scores in purple

def trunk_get_color(score):
    if 1 <= score <= 2:
        return 'green'
    elif 3 <= score <= 4:
        return 'orange'
    elif 5 <= score <= 6:
        return 'red'
    else:
        return 'purple'  # Other scores in purple