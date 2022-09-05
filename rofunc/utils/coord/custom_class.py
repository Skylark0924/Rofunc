import numpy as np


class Pose(object):
    def __init__(self, position: np.ndarray, orientation: np.ndarray):
        """

        Args:
            position: n x 3 array, [[x, y, z]]
            orientation: n x 4 array, [[w, x, y, z]]
        """
        assert len(position) == len(orientation)
        assert position.shape[1] == 3
        assert orientation.shape[1] == 4

        self.position = position
        self.orientation = orientation
        self.pose = np.vstack((position, orientation))

    def __len__(self):
        return self.pose.shape[0]
