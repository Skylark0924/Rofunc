import os
import numpy as np
import matplotlib.pyplot as plt
import pbdlib as pbd
from scipy.io import loadmat  # loading data from matlab

# <editor-fold desc="Learning Hidden Markov Model (HMM)">
letter = 'Z'  # choose a letter in the alphabet

datapath = os.path.dirname(pbd.__file__) + '/data/2Dletters/'
data = loadmat(datapath + '%s.mat' % letter)

demos_x = [d['pos'][0][0].T for d in data['demos'][0]]  # Position data
demos_dx = [d['vel'][0][0].T for d in data['demos'][0]]  # Velocity data
demos_xdx = [np.hstack([_x, _dx]) for _x, _dx in zip(demos_x, demos_dx)]  # Position-velocity

model = pbd.HMM(nb_states=3, nb_dim=4)

model.init_hmm_kbins(demos_xdx)  # initializing model

# EM to train model
model.em(demos_xdx, reg=1e-3)

# plotting
fig, ax = plt.subplots(ncols=3)
fig.set_size_inches(12, 3.5)

# position plotting
ax[0].set_title('pos')
for p in demos_x:
    ax[0].plot(p[:, 0], p[:, 1])

pbd.plot_gmm(model.mu, model.sigma, ax=ax[0], dim=[0, 1]);

# velocity plotting
ax[1].set_title('vel')
for p in demos_dx:
    ax[1].plot(p[:, 0], p[:, 1])

pbd.plot_gmm(model.mu, model.sigma, ax=ax[1], dim=[2, 3]);

# plotting transition matrix
ax[2].set_title('transition')
ax[2].imshow(np.log(model.Trans + 1e-10), interpolation='nearest', vmin=-5, cmap='viridis');
plt.tight_layout()
# plt.show()
# </editor-fold>


# <editor-fold desc="Reproduction (LQR)">
# <editor-fold desc="Get sequence of states ">
demo_idx = 2
sq = model.viterbi(demos_xdx[demo_idx])

plt.figure(figsize=(5, 1))
# plt.axis('off')
plt.plot(sq, lw=3)
plt.xlabel('timestep')
# plt.show()
# </editor-fold>

# <editor-fold desc="Create and solve LQR">
A, b = pbd.utils.get_canonical(2, 2, 0.01)
lqr = pbd.LQR(A, b, horizon=demos_xdx[demo_idx].shape[0])
lqr.gmm_xi = model, sq
lqr.gmm_u = -4.
lqr.ricatti()
lqr._v[0].shape
xi, _ = lqr.get_seq(demos_xdx[demo_idx][0])
# </editor-fold>

# <editor-fold desc="Plotting reproduced trajectory (position and velocity)">
fig, ax = plt.subplots(ncols=2)
fig.set_size_inches(16, 8)

# position plotting
ax[0].set_title('position')
for p in demos_x:
    ax[0].plot(p[:, 0], p[:, 1], alpha=0.4)
pbd.plot_gmm(model.mu, model.sigma, ax=ax[0], dim=[0, 1])

ax[0].plot(xi[:, 0], xi[:, 1], 'b', lw=3)
ax[0].plot(lqr.ds[:, 0], lqr.ds[:, 1], 'gold', lw=3)

# velocity plotting
ax[1].set_title('velocity')
for p in demos_dx:
    ax[1].plot(p[:, 0], p[:, 1], alpha=0.4)

ax[1].plot(xi[:, 2], xi[:, 3], 'b', lw=3, label='repro')
ax[1].plot(lqr.ds[:, 2], lqr.ds[:, 3], 'gold', lw=3)

plt.legend()
pbd.plot_gmm(model.mu, model.sigma, ax=ax[1], dim=[2, 3])
plt.show()
# </editor-fold>
# </editor-fold>
