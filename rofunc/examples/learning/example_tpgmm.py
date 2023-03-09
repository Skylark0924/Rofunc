import numpy as np
import rofunc as rf

raw_demo = np.load('/home/ubuntu/Downloads/010-003/LeftHand.npy')
raw_demo = np.expand_dims(raw_demo, axis=0)
demos_x = np.vstack((raw_demo[:, 430:525, :], raw_demo[:, 240:335, :], raw_demo[:, 335:430, :]))

representation = rf.lfd.tpgmm.TPGMM(demos_x)
model = representation.fit(plot=True)
# traj = representation.reproduce(model, show_demo_idx=2, plot=True)
traj = representation.generate(model, plot=True)
