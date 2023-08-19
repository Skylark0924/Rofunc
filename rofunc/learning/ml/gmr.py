# Copyright 2023, Junjia LIU, jjliu@mae.cuhk.edu.hk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np
import pbdlib as pbd

from rofunc.learning.ml.gmm import GMM
from rofunc.utils.visualab.distribution import gmm_plot


class GMR:
    def __init__(self, demos_x, demos_dx=None, demos_xdx=None, nb_states=4, reg=1e-3, horizon=100, plot=False):
        self.demos_x = demos_x
        self.demos_dx = demos_dx
        self.demos_xdx = demos_xdx
        self.nb_states = nb_states
        self.reg = reg
        self.horizon = horizon
        self.plot = plot

        self.t = self.traj_align()
        self.demos = [np.hstack([self.t[:, None], d]) for d in self.demos_xdx]

    def traj_align(self):
        self.demos_x, self.demos_dx, self.demos_xdx = pbd.utils.align_trajectories(self.demos_x,
                                                                                   [self.demos_dx, self.demos_xdx])

        t = np.linspace(0, self.horizon, self.demos_x[0].shape[0])
        if self.plot:
            fig, ax = plt.subplots(nrows=2)
            for d in self.demos_x:
                ax[0].set_prop_cycle(None)
                ax[0].plot(d)

            ax[1].plot(t)
            plt.show()
        return t

    def gmm_learning(self):
        model = GMM(nb_states=self.nb_states)
        model.init_hmm_kbins(self.demos)  # initializing model

        data = np.vstack([d for d in self.demos])
        model.em(data, reg=self.reg)

        # plotting
        if self.plot:
            fig, ax = plt.subplots(nrows=4)
            fig.set_size_inches(12, 7.5)

            # position plotting
            for i in range(4):
                for p in self.demos:
                    ax[i].plot(p[:, 0], p[:, i + 1])

                gmm_plot(model.mu, model.sigma, ax=ax[i], dim=[0, i + 1])
            plt.show()
        return model

    def estimate(self, model, cond_input, dim_in, dim_out):
        mu, sigma = model.condition(cond_input, dim_in=dim_in, dim_out=dim_out)

        if self.plot:
            gmm_plot(mu, sigma, dim=[0, 1], color='orangered', alpha=0.3)
            for d in self.demos_x:
                plt.plot(d[:, 0], d[:, 1])
            plt.show()
        return mu, sigma

    def fit(self):
        model = self.gmm_learning()
        mu, sigma = self.estimate(model, self.t[:, None], dim_in=slice(0, 1), dim_out=slice(1, 5))
        return model, mu, sigma
