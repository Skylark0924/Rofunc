import numpy as np

from .hmm import *
from .functions import *
from .model import *


class OnlineForwardVariable():
	def __init__(self):
		self.nbD = None
		self.bmx = None
		self.ALPHA = None
		self.S = None
		self.h = None
		pass


class HSMM(HMM):
	def __init__(self, nb_states=2, nb_dim=1):
		HMM.__init__(self, nb_states, nb_dim)
		self.Trans_Fw = np.zeros((self.nb_dim, self.nb_states))

		self._mu_d = None
		self._sigma_d = None
		self._trans_d = None

	@property
	def trans_d(self):
		return self._trans_d

	@trans_d.setter
	def trans_d(self, value):
		self._trans_d = value

	@property
	def mu_d(self):
		return self._mu_d

	@mu_d.setter
	def mu_d(self, value):
		self._mu_d = value

	@property
	def sigma_d(self):
		return self._sigma_d

	@sigma_d.setter
	def sigma_d(self, value):
		self._sigma_d = value

	def make_finish_state(self, demos, dep_mask=None):
		state_sequ = [np.array(self.viterbi(d)) for d in demos]
		for i, d in enumerate(demos):
			state_sequ[i][-3:] = self.nb_states
		super(HSMM, self).make_finish_state(demos, dep_mask)
		self.compute_duration(sequ=state_sequ)

	def compute_duration(self, demos=None, dur_reg=2.0, marginal=None, sequ=None, last=True):
		"""
		Empirical computation of HSMM parameters based on counting transition and durations

		:param demos:	[list of np.array([nb_timestep, nb_dim])]
		:return:
		"""
		# reformat transition matrix: By removing self transition
		# self.Trans_Pd = self.Trans - np.diag(np.diag(self.Trans)) + realmin
		# self.Trans_Pd /= colvec(np.sum(self.Trans_Pd, axis=1))

		# init duration components
		self.Mu_Pd = np.zeros(self.nb_states)
		self.Sigma_Pd = np.zeros(self.nb_states)

		# reconstruct sequence of states from all demonstrations
		state_seq = []

		trans_list = np.zeros((self.nb_states, self.nb_states))  # create a table to count the transition

		s = demos if demos is not None else sequ
		# reformat transition matrix by counting the transition
		for j, d in enumerate(s):
			if sequ is None:
				state_seq_tmp = self.viterbi(d) if marginal is None else self.viterbi(d[:, marginal])
			else:
				state_seq_tmp = d.tolist()
			prev_state = 0

			for i, state in enumerate(state_seq_tmp):

				if i == 0:  # first state of sequence :
					pass
				elif i == len(state_seq_tmp) - 1 and last:  # last state of sequence
					trans_list[state][state] += 1.0
				elif state != prev_state:  # transition detected
					trans_list[prev_state][state] += 1.0

				prev_state = state
			# make a list of state to compute the durations
			state_seq += state_seq_tmp

		self.Trans_Pd = trans_list
		# make sum to one
		for i in range(self.nb_states):
			sum = np.sum(self.Trans_Pd[i, :])
			if sum > realmin:
				self.Trans_Pd[i, :] /= sum

		# print state_seq

		# list of duration
		stateDuration = [[] for i in range(self.nb_states)]

		currState = state_seq[0]
		cnt = 1

		for i, state in enumerate(state_seq):
			if i == len(state_seq) - 1:  # last state of sequence
				stateDuration[currState] += [cnt]
			elif state == currState:
				cnt += 1
			else:
				stateDuration[currState] += [cnt]
				cnt = 1
				currState = state

		# print stateDuration
		for i in range(self.nb_states):
			self.Mu_Pd[i] = np.mean(stateDuration[i])
			if len(stateDuration[i]) > 1:
				self.Sigma_Pd[i] = np.std(stateDuration[i]) + dur_reg
			else:
				self.Sigma_Pd[i] = dur_reg

	def em(self, demos, **kwargs):
		gamma = HMM.em(self, demos, **kwargs)

		self.compute_duration(demos)
		return gamma

	def compute_messages(self, demo=None, dep=None, table=None, marginal=None, sample_size=200, p0=None):
		if isinstance(demo, dict):
			sample_size = demo['x'].shape[0]

		if marginal == []:
			alpha, beta, gamma, zeta, c = self.forward_variable_ts(sample_size, p0=p0), None, None, None, None
		else:
			alpha, beta, gamma, zeta, c = self.forward_variable(sample_size, demo, marginal), None, None, None, None

		return alpha, beta, gamma, zeta, c

	def forward_variable_ts(self, n_step, p0=None):
		"""
		Compute forward variables without any observation of the sequence.

		:param n_step: 			int
			Number of step for forward variable computation
		:return:
		"""

		nbD = np.round(4 * n_step // self.nb_states)

		self.Pd = np.zeros((self.nb_states, nbD))
		# Precomputation of duration probabilities

		for i in range(self.nb_states):
			self.Pd[i, :] = multi_variate_normal(np.arange(nbD), self.Mu_Pd[i], self.Sigma_Pd[i], log=False)
			self.Pd[i, :] = self.Pd[i, :] / np.sum(self.Pd[i, :])

		h = np.zeros((self.nb_states, n_step))

		ALPHA, S, h[:, 0] = self._fwd_init_ts(nbD, p0=p0)

		for i in range(1, n_step):
			ALPHA, S, h[:, i] = self._fwd_step_ts(ALPHA, S, nbD)

		h /= np.sum(h, axis=0)
		return h

	def _fwd_init_ts(self, nbD, p0=None):
		"""
		Initiatize forward variable computation based only on duration (no observation)
		:param nbD: number of time steps
		:return:
		"""
		if p0 is None:
			ALPHA = np.tile(self.init_priors, [nbD, 1]).T * self.Pd
		else:
			ALPHA = np.tile(p0, [nbD, 1]).T * self.Pd

		S = np.dot(self.Trans_Pd.T, ALPHA[:, [0]])  # use [idx] to keep the dimension

		return ALPHA, S, np.sum(ALPHA, axis=1)

	def _fwd_step_ts(self, ALPHA, S, nbD):
		"""
		Step of forward variable computation based only on duration (no observation)
		:return:
		"""
		ALPHA = np.concatenate((S[:, [-1]] * self.Pd[:, 0:nbD - 1] + ALPHA[:, 1:nbD],
								S[:, [-1]] * self.Pd[:, [nbD - 1]]), axis=1)

		S = np.concatenate((S, np.dot(self.Trans_Pd.T, ALPHA[:, [0]])), axis=1)

		return ALPHA, S, np.sum(ALPHA, axis=1)

	def forward_variable(self, n_step=None, demo=None, marginal=None, dep=None, p_obs=None):
		"""
		Compute the forward variable with some observations

		:param demo: 	[np.array([nb_timestep, nb_dim])]
		:param dep: 	[A x [B x [int]]] A list of list of dimensions
			Each list of dimensions indicates a dependence of variables in the covariance matrix
			E.g. [[0],[1],[2]] indicates a diagonal covariance matrix
			E.g. [[0, 1], [2]] indicates a full covariance matrix between [0, 1] and no
			covariance with dim [2]
		:param table: 	np.array([nb_states, nb_demos]) - composed of 0 and 1
			A mask that avoid some demos to be assigned to some states
		:param marginal: [slice(dim_start, dim_end)] or []
			If not None, compute messages with marginals probabilities
			If [] compute messages without observations, use size
			(can be used for time-series regression)
		:param p_obs: 		np.array([nb_states, nb_timesteps])
				custom observation probabilities
		:return:
		"""
		if isinstance(demo, np.ndarray):
			n_step = demo.shape[0]
		elif isinstance(demo, dict):
			n_step = demo['x'].shape[0]

		nbD = np.round(4 * n_step // self.nb_states)
		if nbD == 0:
			nbD = 10
		self.Pd = np.zeros((self.nb_states, nbD))
		# Precomputation of duration probabilities
		for i in range(self.nb_states):
			self.Pd[i, :] = multi_variate_normal(np.arange(nbD), self.Mu_Pd[i], self.Sigma_Pd[i], log=False)
			self.Pd[i, :] = self.Pd[i, :] / (np.sum(self.Pd[i, :])+realmin)
		# compute observation marginal probabilities
		p_obs, _ = self.obs_likelihood(demo, dep, marginal, n_step)

		self._B = p_obs

		h = np.zeros((self.nb_states, n_step))
		bmx, ALPHA, S, h[:, 0] = self._fwd_init(nbD, p_obs[:, 0])

		for i in range(1, n_step):
			bmx, ALPHA, S, h[:, i] = self._fwd_step(bmx, ALPHA, S, nbD, p_obs[:, i])

		h /= np.sum(h, axis=0)

		return h

	def _fwd_init(self, nbD, priors):
		"""

		:param nbD:
		:return:
		"""
		bmx = np.zeros((self.nb_states, 1))

		Btmp = priors
		ALPHA = np.tile(self.init_priors, [nbD, 1]).T * self.Pd

		# r = Btmp.T * np.sum(ALPHA, axis=1)
		r = np.dot(Btmp.T, np.sum(ALPHA, axis=1))

		bmx[:, 0] = Btmp / r
		E = bmx * ALPHA[:, [0]]
		S = np.dot(self.Trans_Pd.T, E)  # use [idx] to keep the dimension

		return bmx, ALPHA, S, Btmp * np.sum(ALPHA, axis=1)

	def _fwd_step(self, bmx, ALPHA, S, nbD, obs_marginal=None):
		"""

		:param bmx:
		:param ALPHA:
		:param S:
		:param nbD:
		:return:
		"""

		Btmp = obs_marginal

		ALPHA = np.concatenate((S[:, [-1]] * self.Pd[:, 0:nbD - 1] + bmx[:, [-1]] * ALPHA[:, 1:nbD],
								S[:, [-1]] * self.Pd[:, [nbD - 1]]), axis=1)

		r = np.dot(Btmp.T, np.sum(ALPHA, axis=1))
		bmx = np.concatenate((bmx, Btmp[:, None] / r), axis=1)
		E = bmx[:, [-1]] * ALPHA[:, [0]]

		S = np.concatenate((S, np.dot(self.Trans_Pd.T, ALPHA[:, [0]])), axis=1)
		alpha = Btmp * np.sum(ALPHA, axis=1)
		alpha /= np.sum(alpha)
		return bmx, ALPHA, S, alpha

	########################################################################################
	## SANDBOX ABOVE
	########################################################################################

	@property
	def Sigma_Pd(self):
		return self.sigma_d

	@Sigma_Pd.setter
	def Sigma_Pd(self, value):
		self.sigma_d = value

	@property
	def Mu_Pd(self):
		return self.mu_d

	@Mu_Pd.setter
	def Mu_Pd(self, value):
		self.mu_d = value

	@property
	def Trans_Pd(self):
		return self.trans_d

	@Trans_Pd.setter
	def Trans_Pd(self, value):
		self.trans_d = value

	def forward_variable_priors(self, n_step, priors, tp_param=None, start_priors=None):
		"""
		Compute the forward variable with some priors over the states

		:param n_step: 			[int]
			Number of step for forward variable computation
		:param priors: 			[np.array((N,))]
			Priors over the states
		:param start_priors: 	[np.array((N,))]
			Priors for localizing at first step

		:return:
		"""
		if tp_param is None:
			try:
				self.Trans_Fw = self.tp_trans.Prior_Trans
			except:
				# print "No task-parametrized transition matrix : normal transition matrix will be used"
				self.Trans_Fw = self.Trans_Pd
		else:  # compute the transition matrix for current parameters
			self._update_transition_matrix(tp_param)

		# nbD = np.round(2 * n_step/self.nb_states)
		nbD = np.round(2 * n_step)

		self.Pd = np.zeros((self.nb_states, nbD))

		# Precomputation of duration probabilities
		for i in range(self.nb_states):
			self.Pd[i, :] = multi_variate_normal(np.arange(nbD), self.Mu_Pd[i], self.Sigma_Pd[i])
			if np.sum(self.Pd[i, :]) < 1e-50:
				self.Pd[i, :] = 1.0 / self.Pd[i, :].shape[0]
			else:
				self.Pd[i, :] = self.Pd[i, :] / np.sum(self.Pd[i, :])

		h = np.zeros((self.nb_states, n_step))

		priors = colvec(priors)
		priors /= np.sum(priors)

		bmx, ALPHA, S, h[:, [0]] = self._fwd_init_priors(nbD, priors, start_priors=start_priors)

		for i in range(1, n_step):
			bmx, ALPHA, S, h[:, [i]] = self._fwd_step_priors(bmx, ALPHA, S, nbD, priors)

		h /= np.sum(h, axis=0)

		return h

	def online_forward_variable_prob(self, n_step, priors, tp_param=None, start_priors=None, nb_sum=None):
		"""

		:param n_step:			[int]
		:param priors: 			[np.array((nb_states,))]
		:param tp_param: 		[np.array((nb_input_dim,))]
		:param start_priors:	[np.array((nb_states,))]
		:return:
		"""
		if tp_param is None:
			try:
				# self.Trans_Fw = self.tp_trans.Prior_Trans
				self.Trans_Fw = self.Trans_Pd
			# print self.Trans_Fw
			except:
				print("No task-parametrized transition matrix : normal transition matrix will be used")
				self.Trans_Fw = self.Trans_Pd
			# print self.Trans_Fw
		else:  # compute the transition matrix for current parameters
			self._update_transition_matrix(tp_param)

		self.ol = OnlineForwardVariable()

		# self.ol.nbD = np.round(2 * n_step / self.nb_states)
		if nb_sum is None:
			self.ol.nbD = np.round(2 * n_step)
		else:
			self.ol.nbD = nb_sum

		self.Pd = np.zeros((self.nb_states, self.ol.nbD))

		# Precomputation of duration probabilities
		for i in range(self.nb_states):
			self.Pd[i, :] = multi_variate_normal(np.arange(self.ol.nbD), self.Mu_Pd[i],
												 self.Sigma_Pd[i], log=False)
			self.Pd[i, :] = self.Pd[i, :] / np.sum(self.Pd[i, :])

		h = np.zeros((self.nb_states, n_step))

		self.ol.h = np.zeros((self.nb_states, 1))

		priors = colvec(priors)
		priors /= np.sum(priors)

		self.ol.bmx, self.ol.ALPHA, self.ol.S, self.ol.h = self._fwd_init_priors(self.ol.nbD, priors,
																				 start_priors=start_priors)

		# for i in range(1, n_step):
		# 	bmx, ALPHA, S, h[:, [i]] = self._fwd_step_priors(bmx, ALPHA, S, self.ol.nbD, priors)
		#
		# h /= np.sum(h, axis=0)

		return self.ol.h

	def online_forward_variable_prob_step(self, priors):
		"""
		Single step to compute an online forward variable for HSMM.

		:param priors: 			[np.array((nb_states,))]
		:return:
		"""

		priors = colvec(priors)
		try:
			self.ol.bmx, self.ol.ALPHA, self.ol.S, self.ol.h = \
				self._fwd_step_priors(self.ol.bmx, self.ol.ALPHA, self.ol.S, self.ol.nbD, priors,
									  trans_reg=0.00, trans_diag=0.00)
			return self.ol.h
		except:
			# traceback.print_exc(file=sys.stdout)
			return None

	def online_forward_variable_prob_predict(self, n_step, priors):
		"""
		Compute prediction for n_step timestep on the current online forward variable.

		:param priors: 			[np.array((nb_states,))]
		:return:
		"""

		h = np.zeros((self.nb_states, n_step))

		priors = colvec(priors)
		priors /= np.sum(priors)

		# bmx, ALPHA, S, h[:, [0]] = self._fwd_init_priors(nbD, priors, start_priors=start_priors)
		h[:, [0]] = self.ol.h
		bmx = self.ol.bmx
		ALPHA = self.ol.ALPHA
		S = self.ol.S

		try:
			for i in range(1, n_step):
				bmx, ALPHA, S, h[:, [i]] = self._fwd_step_priors(bmx, ALPHA, S, self.ol.nbD, priors)

		except:
			h = np.tile(self.ol.h, (1, n_step))

		# traceback.print_exc(file=sys.stdout)

		h /= np.sum(h, axis=0)

		return h

	def forward_variable_hsum(self, n_step, Data, tp_param=None):
		"""
		Compute the forward variable with observation

		:param n_step: 		int
			Number of step for forward variable computation
		:param priors: 		np.array((N,))
			Priors over the states

		:return:
		"""
		if tp_param is None:
			# self.Trans_Fw = self.tp_trans.Prior_Trans
			self.Trans_Fw = self.Trans_Pd
		else:  # compute the transition matrix for current parameters
			self._update_transition_matrix(tp_param)

		# nbD = np.round(2 * n_step/self.nb_states)
		nbD = np.round(4 * n_step)

		self.Pd = np.zeros((self.nb_states, nbD))

		# Precomputation of duration probabilities
		for i in range(self.nb_states):
			self.Pd[i, :] = multi_variate_normal(np.arange(nbD), self.Mu_Pd[i], self.Sigma_Pd[i])
			self.Pd[i, :] = self.Pd[i, :] / np.sum(self.Pd[i, :])

		if np.isnan(self.Pd).any():
			print("Problem of duration probabilities")
			return

		h = np.zeros((self.nb_states, n_step))

		bmx, ALPHA, S, h[:, [0]] = self._fwd_init_hsum(nbD, Data[:, 1])
		for i in range(1, n_step):
			bmx, ALPHA, S, h[:, [i]] = self._fwd_step_hsum(bmx, ALPHA, S, nbD, Data[:, i])

		h /= np.sum(h, axis=0)

		return h

	def _fwd_init_priors(self, nbD, priors, start_priors=None):
		"""

		:param nbD:
		:return:
		"""
		bmx = np.zeros((self.nb_states, 1))

		Btmp = priors

		if start_priors is None:
			ALPHA = np.tile(colvec(self.init_priors), [1, nbD]) * self.Pd
		else:
			ALPHA = np.tile(colvec(start_priors), [1, nbD]) * self.Pd
		# r = Btmp.T * np.sum(ALPHA, axis=1)
		r = np.dot(Btmp.T, np.sum(ALPHA, axis=1))

		bmx[:, [0]] = Btmp / r
		E = bmx * ALPHA[:, [0]]
		S = np.dot(self.Trans_Fw.T, E)  # use [idx] to keep the dimension

		return bmx, ALPHA, S, Btmp * colvec(np.sum(ALPHA, axis=1))

	def _fwd_step_priors(self, bmx, ALPHA, S, nbD, priors, trans_reg=0.0, trans_diag=0.0):
		"""

		:param bmx:
		:param ALPHA:
		:param S:
		:param nbD:
		:return:
		"""

		Btmp = priors

		ALPHA = np.concatenate((S[:, [-1]] * self.Pd[:, 0:nbD - 1] + bmx[:, [-1]] * ALPHA[:, 1:nbD],
								S[:, [-1]] * self.Pd[:, [nbD - 1]]), axis=1)

		r = np.dot(Btmp.T, np.sum(ALPHA, axis=1))
		bmx = np.concatenate((bmx, Btmp / r), axis=1)
		E = bmx[:, [-1]] * ALPHA[:, [0]]

		S = np.concatenate(
			(S, np.dot(self.Trans_Fw.T + np.eye(self.nb_states) * trans_diag + trans_reg, ALPHA[:, [0]])), axis=1)
		alpha = Btmp * colvec(np.sum(ALPHA, axis=1))
		alpha /= np.sum(alpha)
		return bmx, ALPHA, S, alpha

	def _fwd_init_hsum(self, nbD, Data):
		"""

		:param nbD:
		:return:
		"""
		bmx = np.zeros((self.nb_states, 1))

		Btmp = np.zeros((self.nb_states, 1))

		for i in range(self.nb_states):
			Btmp[i] = multi_variate_normal(Data.reshape(-1, 1), self.Mu[:, i], self.Sigma[:, :, i]) + 1e-12

		Btmp /= np.sum(Btmp)

		ALPHA = np.tile(colvec(self.init_priors), [1, nbD]) * self.Pd
		# r = Btmp.T * np.sum(ALPHA, axis=1)
		r = np.dot(Btmp.T, np.sum(ALPHA, axis=1))

		bmx[:, [0]] = Btmp / r
		E = bmx * ALPHA[:, [0]]
		S = np.dot(self.Trans_Fw.T, E)  # use [idx] to keep the dimension

		return bmx, ALPHA, S, Btmp * colvec(np.sum(ALPHA, axis=1))

	def _fwd_step_hsum(self, bmx, ALPHA, S, nbD, Data):
		"""

		:param bmx:
		:param ALPHA:
		:param S:
		:param nbD:
		:return:
		"""

		Btmp = np.zeros((self.nb_states, 1))

		for i in range(self.nb_states):
			Btmp[i] = multi_variate_normal(Data.reshape(-1, 1), self.Mu[:, i], self.Sigma[:, :, i]) + 1e-12

		Btmp /= np.sum(Btmp)

		ALPHA = np.concatenate((S[:, [-1]] * self.Pd[:, 0:nbD - 1] + bmx[:, [-1]] * ALPHA[:, 1:nbD],
								S[:, [-1]] * self.Pd[:, [nbD - 1]]), axis=1)

		r = np.dot(Btmp.T, np.sum(ALPHA, axis=1))
		bmx = np.concatenate((bmx, Btmp / r), axis=1)
		E = bmx[:, [-1]] * ALPHA[:, [0]]

		S = np.concatenate((S, np.dot(self.Trans_Fw.T, ALPHA[:, [0]])), axis=1)
		alpha = Btmp * colvec(np.sum(ALPHA, axis=1))
		alpha /= np.sum(alpha)
		return bmx, ALPHA, S, alpha
