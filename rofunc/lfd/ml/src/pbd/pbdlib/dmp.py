import numpy as np
import matplotlib.pyplot as plt
from .gmm import GMM
from .hmm import HMM
from .hsmm import HSMM
from .mtmm import VBayesianGMM
from .plot import plot_gmm
from scipy.stats import multivariate_normal


class DMP(object):
    """
	Dynamic Movement Primitives with shaping function f(.) learned by Locally Weighted Regression (LWR) or
	Bayesian Gaussian Mixture Model (BGMM).

	"""

    def __init__(self, bf_number, K, x, state_dependent=False, mixed=False, formula=1):
        # Assumes the same data_length for each demo of each task
        # also for each task, assumes the same number of demos
        # duration might differ
        # X contains [task_number, demo_number, data_length, [time, features]]
        self.task_number = x.shape[0]
        self.demo_number = x.shape[1]
        self.data_length = x.shape[2]
        self.dof = x.shape[-1] - 1  # Number of degrees of freedom for DMP

        # Making the time start from 0s.
        x_subs = np.zeros_like(x)
        x_subs[:, :, :, 0] = np.tile(x[:, :, 0, 0][:, :, None], (1, 1, x.shape[2]))
        x = x - x_subs

        self.t = x[:, :, :, 0]  # Time information : [task_number, demo_number, data_length]
        self.x = x[:, :, :, 1:]  # Demonstrated trajectory : [task_number, demo_number, data_length, features]

        # Initial position : [task_number, demo_number, data_length, features]
        self.x0 = np.tile(self.x[:, :, 0, :][:, :, None, :], (1, 1, self.data_length, 1))
        # Goal position : [task_number, demo_number, data_length, features]
        self.g = np.tile(self.x[:, :, -1, :][:, :, None, :], (1, 1, self.data_length, 1))

        self.bf_number = bf_number  # Basis function number

        self.K = np.diag(K)  # Spring constant
        self.D = np.diag(2 * np.sqrt(K))  # Damping constant, critically damped

        # Integrating the canonical system and mapping 's' to the time
        self.tau = self.t[:, :, -1]  # Duration of the demonstrations : [task_number, demo_number]
        convergence_rate = 0.01
        self.alpha = -np.log(convergence_rate)

        # self.s : [task_number, demo_number, data_length, 1]
        self.s = np.exp(
            np.einsum('td, tdl->tdl', -self.alpha / self.tau, self.t)
        )[:, :, :, None]
        s = np.tile(self.s, (1, 1, 1, self.bf_number))

        # Numerical Differentiations
        dt = np.diff(self.t[0, 0, :], axis=-1)[0]
        x_dot = np.gradient(self.x, dt, axis=-2)  # [task_number, demo_number, data_length, features]
        x_ddot = np.gradient(x_dot, dt, axis=-2)

        v = np.einsum('td,tdlj->tdlj', self.tau, x_dot)
        v_dot = np.einsum('td,tdlj->tdlj', self.tau, x_ddot)
        tau_v_dot = np.einsum('td,tdlj->tdlj', self.tau, v_dot)

        # Finding the target shaping function
        K_inv = np.linalg.inv(self.K)  # [features, features]
        Dv = np.einsum('ij,tdlj->tdli', self.D, v)  # [task_number, demo_number, data_length, features]


        self.formula = formula
        # # Pastor et al, 2011, Learning and Generalization of Motor Skills ... (improved)
        if self.formula == 1:
            self.f_target = np.einsum('ij,tdlj->tdli', K_inv, tau_v_dot + Dv) + \
                            (self.x - self.g) + \
                            np.einsum('tdl,tdlj->tdlj', s[:, :, :, 0], self.g - self.x0)
        elif self.formula == 2:
            # # Heiko Hoffmann form 2009
            self.f_target = np.einsum('ij,tdlj->tdli', K_inv, tau_v_dot + Dv) + \
            				(self.x - self.g) + \
            				np.einsum('tdl,tdlj->tdlj', s[:, :, :, 0], self.g - self.x0)
            self.f_target = self.f_target / self.s
        elif self.formula == 3:
            # # Pastor et al, 2011, Learning and Generalization of Motor Skills ... (classic)
            self.f_target = (tau_v_dot + Dv + np.einsum('ij,tdlj->tdli', self.K, self.x - self.g))/(self.g-self.x0)
        elif self.formula == 4:
            # # The simplest form
            self.f_target = tau_v_dot + Dv + np.einsum('ij,tdlj->tdli', self.K, self.x - self.g)
        elif self.formula == 5:
            # # Sylvain's form
            self.f_target = (tau_v_dot + Dv + np.einsum('ij,tdlj->tdli', self.K, self.x - self.g))/self.s

        if state_dependent:
            self.inp = self.x
        elif mixed:
            self.inp = np.concatenate([self.s, self.x], axis=-1)
        else:
            self.inp = self.s

        self.method = None
        self.state_dependent = state_dependent
        self.mixed = mixed

        self._weights = None

    def learn_lwr(self, f_target=None, bf_number=None):
        if f_target is not None:
            pass
        else:
            assert self.task_number == 1, \
                "Only one task can be learned with LWR," \
                " try learn_contextual_bgmm to learn a parametric task"
            self.method = "lwr"
            f_target = self.f_target

        if not bf_number:
            bf_number = self.bf_number

        task_number = f_target.shape[0]
        demo_number = f_target.shape[1]

        s = np.tile(self.s[0, 0][None, None], (task_number, demo_number, 1, 1))

        # Creating basis functions and psi_matrix
        # Centers logarithmically distributed between 0.001 and 1
        # self.c : [task_number, demo_number, data_length, bf_number]

        self.c_exec = np.logspace(-3, 0, num=bf_number)
        self.h_exec = bf_number / (self.c_exec ** 2)

        c = np.tile(
            self.c_exec[None, None, None],
            (task_number, demo_number, self.data_length, 1)
        )  # centers of basis functions

        h = bf_number / (c ** 2)  # widths of basis functions
        # self.psi_matrix : [task_number, demo_number, data_length, bf_number]
        psi_matrix = np.exp(-h * (s - c) ** 2)

        # self.inv_sum_bfs : [task_number, demo_number, data_length]
        inv_sum_bfs = 1.0 / np.sum(psi_matrix, axis=-1)

        bf_target = np.einsum('tdlb,tdl->tdlb', psi_matrix * s, inv_sum_bfs)

        sol = np.linalg.lstsq(
            np.concatenate(np.concatenate(bf_target, axis=0), axis=0),
            np.concatenate(np.concatenate(f_target, axis=0), axis=0),
        rcond=None)
        self._weights = sol[0]

    def learn_gmm(self, n_comp=20, hmm=False, hsmm=False, plot=False, reg=1e-5):
        x_joint = np.concatenate([
            np.concatenate(self.inp, axis=0),
            np.concatenate(self.f_target, axis=0),
        ], axis=-1)

        self.n_joint = x_joint.shape[-1]
        self.inp_dim = self.inp.shape[-1]
        self.x_joint = x_joint

        if hmm:
            self.joint_model = HMM(nb_states=self.bf_number)
            self.method = "HMM"
        elif hsmm:
            self.joint_model = HSMM(nb_states=self.bf_number)
            self.method = "HSMM"
        else:
            self.joint_model = GMM(nb_states=self.bf_number)
            self.method = "GMM"

        self.joint_model.init_hmm_kbins(x_joint)

        x_joint_hmm = x_joint
        x_joint = np.concatenate(x_joint)
        if not hmm and not hsmm:
            self.joint_model.em(x_joint, reg=reg)
        else:
            self.joint_model.em(x_joint_hmm, reg=reg, nb_max_steps=100)

        if plot:
            fig, ax = plt.subplots(nrows=self.n_joint - 1, ncols=self.n_joint - 1, figsize=(10, 10))

            for i in range(self.n_joint):
                for j in range(self.n_joint):
                    if not i == j and j > i:
                        ax[i][j - 1].plot(x_joint[:, i], x_joint[:, j], 'kx')
                        ax[i][j - 1].autoscale(False)
                        plot_gmm(self.joint_model.mu, self.joint_model.sigma, dim=[i, j], ax=ax[i][j - 1], alpha=0.2)

    def learn_bgmm(self, cov_prior=None, plot=False, plot_data=False):
        assert self.task_number == 1, \
            "Only one task can be learned with LWR," \
            " try learn_contextual_bgmm to learn a parametric task"
        self.method = "bgmm"

        # x_joint = np.concatenate([
        # 	np.concatenate(np.concatenate(self.inp, axis=0), axis=0)[:, 0][:, None],
        # 	np.concatenate(np.concatenate(self.f_target, axis=0), axis=0)
        # ], axis=-1)

        x_joint = np.concatenate([
            np.reshape(self.inp, (-1, self.inp.shape[-1])),
            np.reshape(self.f_target, (-1, self.f_target.shape[-1])),
        ], axis=-1)

        self.n_joint = x_joint.shape[1]
        self.inp_dim = self.inp.shape[-1]
        self.x_joint = x_joint

        # self.joint_model = VBayesianGMM({
        # 	'n_components': self.bf_number, 'n_init': 1, 'reg_covar': 1E-6,
        # 	'covariance_prior': cov * 1, 'mean_precision_prior': 20. ** -2,
        # 	'weight_concentration_prior_type': 'dirichlet_process', 'weight_concentration_prior': 1e3,
        # 	'degrees_of_freedom_prior': self.n_joint - 1. + 0.3, 'warm_start': False})

        self.joint_model = VBayesianGMM({
            'n_components': self.bf_number, 'n_init': 5, 'covariance_prior': cov_prior})
        self.joint_model.posterior(data=x_joint)

        if plot:
            print("Used state number: ", self.joint_model.get_used_states().nb_states, "out of ", self.bf_number)
            fig, ax = plt.subplots(nrows=self.n_joint - 1, ncols=self.n_joint - 1, figsize=(10, 10))

            for i in range(self.n_joint):
                for j in range(self.n_joint):
                    if not i == j and j > i:
                        ax[i][j - 1].plot(x_joint[:, i], x_joint[:, j], 'kx')
                        ax[i][j - 1].autoscale(False)
                        self.joint_model.get_used_states().plot(dim=[i, j], ax=ax[i][j - 1], alpha=0.2)

    def learn_contextual_bgmm(self, params, cov_prior=None, plot=False):
        assert self.task_number > 1, \
            "If only one task is to be trained," \
            " try learn_bgmm to learn a non-parametric task"
        self.method = "contextual bgmm"

        x_joint = np.concatenate([
            np.concatenate(np.concatenate(self.s, axis=0), axis=0)[:, 0][:, None],
            np.kron(params, np.ones(int(self.data_length * self.demo_number)))[:, None],
            np.concatenate(np.concatenate(self.f_target, axis=0), axis=0)
        ], axis=-1)

        self.n_joint = x_joint.shape[1]
        self.x_joint = x_joint

        # self.joint_model = VBayesianGMM({
        # 	'n_components': self.bf_number, 'n_init': 5, 'reg_covar': 1E-8,
        # 	'covariance_prior': cov * 1, 'mean_precision_prior': 20. ** -2,
        # 	'weight_concentration_prior_type': 'dirichlet_process', 'weight_concentration_prior': 1e3,
        # 	'degrees_of_freedom_prior': self.n_joint - 1. + 0.3, 'warm_start': False})
        self.joint_model = VBayesianGMM({
            'n_components': self.bf_number, 'n_init': 5, 'covariance_prior': cov_prior})
        self.joint_model.posterior(data=x_joint)

        if plot:
            print("Used state number: ",
                  self.joint_model.get_used_states().nb_states,
                  "out of ", self.bf_number)
            fig, ax = plt.subplots(nrows=self.n_joint - 1, ncols=self.n_joint - 1, figsize=(10, 10))

            for i in range(self.n_joint):
                for j in range(self.n_joint):
                    if not i == j and j > i:
                        ax[i][j - 1].plot(x_joint[:, i], x_joint[:, j], 'kx')
                        # joint_model.get_used_states().plot(dim=[i, j], ax=ax[i][j-1], alpha=0.2)
                        ax[i][j - 1].autoscale(False)
                        self.joint_model.get_used_states().plot(dim=[i, j], ax=ax[i][j - 1], alpha=0.2)
                # joint_model.models[1].get_used_states().plot(dim=[i, j], ax=ax[i][j-1], alpha=0.2, color='b')

    def learn_promp(self, params=None, n_comp=None, cov_prior=None):

        self.method = "contextual_promp"
        weights = np.zeros([self.task_number, self.demo_number, self.bf_number, self.x.shape[-1]])
        for i in range(self.task_number):
            for j in range(self.demo_number):
                self.learn_lwr(self.f_target[i, j][None, None])
                weights[i, j] = self.weights

        if params is None:
            x_joint = np.concatenate(weights, axis=0).reshape([int(self.task_number * self.demo_number), -1])

        else:
            x_joint = np.concatenate([
                np.kron(params, np.ones(int(self.demo_number)))[:, None],
                np.concatenate(weights, axis=0).reshape([int(self.task_number * self.demo_number), -1])
            ], axis=-1)
        self.x_joint = x_joint
        self.n_joint = x_joint.shape[1]

        self.joint_model = VBayesianGMM({
            'n_components': n_comp, 'n_init': 5, 'covariance_prior': cov_prior})
        self.joint_model.posterior(data=x_joint)
        print("Used state number: ",
              self.joint_model.get_used_states().nb_states,
              "out of ", n_comp)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    def execute(self, time, dt, des_tau, x0, g, x, xdot, zeta=0., param=None):
        # Integrate the canonical system
        s = np.exp(((-self.alpha / des_tau) * time))
        if self.state_dependent:
            inp = x
        elif self.mixed:
            inp = np.concatenate([np.array([s]), x], axis=0)
        else:
            inp = np.array([s])

        # Nonlinear function computation
        if self.method == "lwr":
            # Basis functions for the new state
            psi = np.exp(-self.h_exec * ((s - self.c_exec) ** 2))
            sum_of_bfs = np.sum(psi)
            fs_nom = np.sum(np.einsum('bj,b->bj', self.weights, psi), axis=0)
            fs = (fs_nom / sum_of_bfs) * s
        elif self.method == "bgmm":
            mtmm = self.joint_model.condition(
                inp,
                slice(0, self.inp_dim),
                slice(self.inp_dim, self.n_joint))

            # mtmm = self.joint_model.get_used_states().condition(
            # 	inp,
            # 	slice(0, self.inp_dim),
            # 	slice(self.inp_dim, self.n_joint))

            fs = mtmm.sample(1)[0]
        # fs = mtmm.mu[np.argmax(mtmm.priors)]
        # fs = mtmm.get_matching_gaussian()[0][0]

        elif self.method == "contextual bgmm":
            mtmm = self.joint_model.condition(
                np.array([s, param]),
                slice(0, 2),
                slice(2, self.n_joint)
            )
            fs = mtmm.sample(1)[0]
        # fs = mtmm.get_matching_gaussian()[0][0]
        # fs = mtmm.mu[np.argmax(mtmm.priors)]

        elif self.method == "contextual_promp":
            # if time >= 0.+dt:
            # 	# if self.task_number == 1:
            # 	# 	# arg = np.argmin(np.linalg.norm(self._weights[None] - self.joint_model.mu, axis=-1))
            # 	# 	arg = np.argmax(self.joint_model.log_prob_components(self._weights))
            # 	# 	weights = multivariate_normal.rvs(self.joint_model.mu[arg], self.joint_model.sigma[arg])
            # 	# else:
            # 	# 	mtmm = self.joint_model.condition(
            # 	# 		np.array([param]),
            # 	# 		slice(0, 1),
            # 	# 		slice(1, self.n_joint)
            # 	# 	)
            # 	# 	arg = np.argmin(np.linalg.norm(self._weights[None] - mtmm.mu[:, 0], axis=-1))
            # 	# 	weights = multivariate_normal.rvs(mtmm.mu[arg, 0], mtmm.sigma[arg])
            # 	pass
            # else:
            # 	print("time small")
            # if self.task_number == 1:
            # 	weights = self.joint_model.sample(1)[0]
            # else:
            # 	mtmm = self.joint_model.condition(
            # 		np.array([param]),
            # 		slice(0, 1),
            # 		slice(1, self.n_joint)
            # 	)
            #
            # 	weights = mtmm.sample(1)[0]
            #
            # # self._weights = weights
            # print(weights.shape)

            psi = np.exp(-self.h_exec * ((s - self.c_exec) ** 2))
            sum_of_bfs = np.sum(psi)
            fs_nom = np.sum(np.einsum('bj,b->bj', self.weights, psi), axis=0)
            fs = (fs_nom / sum_of_bfs)

        else:
            # gmm_mu, gmm_sigma = self.joint_model.condition(
            # 	inp[None],
            # 	slice(0, self.inp_dim),
            # 	slice(self.inp_dim, self.n_joint))
            # fs = gmm_mu[0]

            priors, mu, sigma = self.joint_model.condition(
                inp[None],
                slice(0, self.inp_dim),
                slice(self.inp_dim, self.n_joint), return_gmm=True)
            if time > 0.01:
                arg = np.argmin(np.linalg.norm(self.fs[None] - mu[:, 0], axis=-1))
                fs = multivariate_normal.rvs(mu[arg, 0], sigma[arg])
            else:
                gmm = GMM(nb_states=self.joint_model.nb_states)
                gmm.mu = mu[:, 0]
                gmm.sigma = sigma
                gmm.priors = priors[:, 0]
                fs = gmm.sample(1)[0]
            # fs = gmm.mu[np.argmax(gmm.priors)]
            self.fs = fs

        # Scaled Velocity needed
        v = des_tau * xdot

        # Main equations
        if self.formula == 1:
            v_dot = (1.0 / des_tau) * (
                    np.dot(self.K, g - x)
                    - np.dot(self.D, v)
                    - np.dot(self.K, (g - x0)) * s
                    + np.dot(self.K, fs)
                    + zeta
            )
        elif self.formula == 2:
            v_dot = (1.0 / des_tau) * (
                    np.dot(self.K, g - x)
                    - np.dot(self.D, v)
                    - np.dot(self.K, (g - x0)) * s
                    + np.dot(self.K, fs) * s
                    + zeta
            )
        elif self.formula == 3:
            v_dot = (1.0 / des_tau) * (
            		np.dot(self.K, g - x)
            		- np.dot(self.D, v)
            		+ fs.dot(g-x0)
            	)
        elif self.formula == 4:
            v_dot = (1.0 / des_tau) * (
            		np.dot(self.K, g - x)
            		- np.dot(self.D, v)
            		+ fs
            	)
        elif self.formula == 5:
            v_dot = (1.0 / des_tau) * (
            		np.dot(self.K, g - x)
            		- np.dot(self.D, v)
            		+ fs*s
            	)

        v = v + v_dot * dt
        xdot = v / des_tau
        x = x + xdot * dt

        return x, xdot

    def rollout(self, dt, des_tau, x0, g, param=None, i=None):
        time = 0
        x = x0
        x_dot = np.zeros_like(x0)
        x_holder = [x0]
        if self._weights is None:
            if self.task_number == 1:
                # weights = self.joint_model.sample(1)[0]
                # print(weights.shape)
                weights = self.joint_model.mu[0]
            # print(weights)
            else:
                mtmm = self.joint_model.condition(
                    np.array([param]),
                    slice(0, 1),
                    slice(1, self.n_joint)
                )

                weights = mtmm.sample(1)[0]
            # print(mtmm.priors)
            # weights = mtmm.mu[1]
            # print(weights)
            # self.weights = weights.reshape((self.bf_number, -1))

        while time <= des_tau:
            x, x_dot = self.execute(time, dt, des_tau, x0, g, x, x_dot, param=param)
            time += dt
            x_holder.append(x)

        return np.stack(x_holder)
