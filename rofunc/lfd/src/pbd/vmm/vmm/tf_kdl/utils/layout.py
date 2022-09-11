
from enum import IntEnum

class FkLayout(IntEnum):
	x = 0		# only position
	xq = 1		# position - quaternion
	xm = 2		# position - vectorized rotation matrix - order 'C'
	xmv = 4		# position - vectorized rotation matrix - order 'F' - [x_1, x_2, x_3, y_1, ..., z_3]
	f = 3		# frame.
