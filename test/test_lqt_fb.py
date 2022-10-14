import rofunc as rf
import numpy as np
from rofunc.config.get_config import *

# <editor-fold desc="7-dim example">
via_points = np.zeros((3, 14))
via_points[0, :7] = np.array([2, 5, 3, 0, 0, 0, 1])
via_points[1, :7] = np.array([3, 1, 1, 0, 0, 0, 1])
via_points[2, :7] = np.array([5, 4, 1, 0, 0, 0, 1])
rf.lqt.uni_fb(via_points)
# </editor-fold>

# <editor-fold desc="2-dim example">
# TODO: need to modify the definition of state noise
# cfg = get_config("./", "lqt_2d")
# via_points = np.array([[2, 5, 0, 0], [3, 1, 0, 0]])
# rf.lqt.uni_fb(via_points, cfg=cfg)
# </editor-fold>

