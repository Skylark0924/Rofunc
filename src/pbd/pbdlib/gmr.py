# -*- coding: utf-8 -*-
"""
Created on Thu Jul 09 18:43:10 2015

@author: Emmanuel Pignat
"""

import numpy as np
from scipy import linalg
from sklearn import mixture



class GMR():
	"""Gaussian Mixture Regression
	"""

	def __init__(self, gmm, slice=False, use_pybdlib_format=False):
		"""

		:param gmm:
		:param slice:
		:param use_pybdlib_format: 		[bool]
			Choose True if covariance and mean is in pybdlib format and not scipy
		"""
		if use_pybdlib_format:
			self.gmm = mixture.GMM(n_components=gmm.nb_states, covariance_type='full')
			self.gmm.weights_ = gmm.Priors

			self.gmm.covars_ = np.transpose(gmm.Sigma, (2, 0, 1))
			self.gmm.means_ = np.transpose(gmm.Mu, (1, 0))
		else:
			self.gmm = gmm

		if slice:
			self.slice_gmm = mixture.GMM(n_components=self.gmm.n_components,
										 covariance_type='full')

		self.use_pybdlib_ = use_pybdlib_format
		self.InvSigmaInIn = [None] * self.gmm.n_components
		self.InvSigmaOutIn = [None] * self.gmm.n_components

		self.SigmaOutTmp = [None] * self.gmm.n_components

		self.inv_tmp = [None] * self.gmm.n_components

		self.pri_tmp = [None] * self.gmm.n_components

		self.cov_tmp = [None] * self.gmm.n_components

		self.input = None
		self.output = None

	# @profile
	def predict_GMM(self, sample, input, output, variance_type='v', predict=False, norm=False,
					reg=1e-9):

		has_changed = False

		# check if input or output changed
		if self.input != input or self.output != output:
			has_changed = True
			self.input = input
			self.output = output

		sloo = np.ix_(self.output, self.output)
		slii = np.ix_(self.input, self.input)
		sloi = np.ix_(self.output, self.input)
		slio = np.ix_(self.input, self.output)

		prob = np.empty(self.gmm.n_components, dtype=float)
		prob_un = np.empty(self.gmm.n_components, dtype=float) # unnormalized

		# get the probability of the sample to be in each gaussian
		for i in range(self.gmm.n_components):
			prob[i] = self.get_pdf(i, sample, has_changed=has_changed, reg=reg) * self.gmm.weights_[i]
			prob_un[i] = self.get_pdf_un(i, sample, has_changed=has_changed) * self.gmm.weights_[i]

		if norm:
			sum_prob = np.sum(prob)
			# sum_prob = np.sum(prob_un)
			if sum_prob:
				beta = prob / sum_prob
				# beta = prob_un / sum_prob
			else:
				beta = np.ones(prob.shape) / prob.shape[0]
		else:
			# sum_prob = np.sum(prob)
			sum_prob = np.sum(prob_un)
			if sum_prob:
				# beta = prob / sum_prob
				beta = prob_un / sum_prob
			else:
				beta = np.ones(prob.shape) / prob.shape[0]

		# print prob_un
		# print prob

		MuOut = None
		SigmaOut = None

		if predict:
			MuOut = np.zeros(len(self.output), dtype=float)
			SigmaOut = np.zeros((len(self.output), len(self.output)), dtype=float)

		MusOut = np.zeros((self.gmm.n_components, len(self.output)), dtype=float)
		SigmasOut = np.zeros((self.gmm.n_components, len(self.output), len(self.output)),
							 dtype=float)

		# get a slice of the gmm model
		for i in range(self.gmm.n_components):

			Mu = self.gmm.means_[i]
			Sigma = self.gmm.covars_[i]

			if has_changed:
				self.InvSigmaInIn[i] = linalg.inv(Sigma[slii])
				self.InvSigmaOutIn[i] = np.dot(Sigma[sloi], self.InvSigmaInIn[i])

			MuOutTmp = Mu[self.output] + np.dot(self.InvSigmaOutIn[i],
												(sample - Mu[self.input]).T)

			if variance_type == 'full':
				self.SigmaOutTmp[i] = MuOutTmp ** 2 + (Sigma[sloo] - \
													   np.dot(self.InvSigmaOutIn[i], \
															  Sigma[np.ix_(self.input,
																		   self.output)])) ** 2

			else:
				if has_changed:
					self.SigmaOutTmp[i] = Sigma[sloo] - \
										  np.dot(self.InvSigmaOutIn[i], \
												 Sigma[slio])

			MusOut[i, :] = MuOutTmp
			SigmasOut[i, :, :] = self.SigmaOutTmp[i]

			if predict:
				MuOut = MuOut + beta[i] * MuOutTmp

				if variance_type == 'full':
					# see course ML : nonlinearRegression.pdf
					SigmaOut = SigmaOut + beta[i] * self.SigmaOutTmp[i] - (beta[
																			   i] * MuOutTmp) ** 2

				else:
					SigmaOut = SigmaOut + beta[i] * self.SigmaOutTmp[i]

				# create a new gmm from this slice
		out_gmm = mixture.GMM(n_components=self.gmm.n_components, covariance_type='full')

		out_gmm.means_ = MusOut
		out_gmm.covars_ = SigmasOut
		out_gmm.weights_ = beta
		out_gmm.converged_ = True

		return (MuOut, SigmaOut, out_gmm)

	# @profile
	def predict_histogramm(self, sample, input, output, variance_type='full'):
		# X = np.asarray(X, dtype=np.float)
		"""
		if X.ndim == 1:
			X = X[:, np.newaxis]
		if X.shape[0] < self.gmm.n_components:
			raise ValueError(
				'GMM estimation with %s components, but got only %s samples' %
				(self.gmm.n_components, X.shape[0]))
		"""
		# set input and output

		has_changed = False
		if self.input != input or self.output != output:
			has_changed = True
			self.input = input
			self.output = output

		N = 1000
		out_hist = [0, 1]

		histogram = np.zeros((len(out_hist), N), dtype=float)

		span = np.empty((len(out_hist), N), dtype=float)
		for i in out_hist:
			span[i] = np.linspace(-1, 1, N)

		inpCr = np.array(self.input, dtype=np.intp)
		outCr = np.array(self.output, dtype=np.intp)

		'''
		if len(self.output) == 1:
			self.output = self.output[0]

		if len(self.input) == 1:
			self.input = self.input[0]
		'''

		prob = np.empty(self.gmm.n_components, dtype=float)

		# 0.1 ms for get_pdf
		for i in range(self.gmm.n_components):
			prob[i] = self.get_pdf(i, sample, has_changed=has_changed) * self.gmm.weights_[
				i]

		try:
			beta = prob / np.sum(prob)
		except:
			beta = np.ones(prob.shape) / prob.shape[0]

		MuOut = np.zeros(len(self.output), dtype=float)

		MusOut = np.zeros((self.gmm.n_components, len(self.output)), dtype=float)

		SigmaOut = np.zeros((len(self.output), len(self.output)), dtype=float)

		SigmasOut = np.zeros((self.gmm.n_components, len(self.output), len(self.output)),
							 dtype=float)

		for i in range(self.gmm.n_components):
			Mu = self.gmm.means_[i]
			Sigma = self.gmm.covars_[i]

			if has_changed:
				self.InvSigmaInIn[i] = linalg.inv(Sigma[np.ix_(self.input, self.input)])
				self.InvSigmaOutIn[i] = np.dot(Sigma[np.ix_(self.output, self.input)],
											   self.InvSigmaInIn[i])

			"""
			MuOutTmp = Mu[self.output] + np.dot((Sigma[np.ix_(self.output, self.input)] * InvSigmaInIn) * \
										 (sample[0,self.input] - Mu[self.input]).T).T
			"""
			if sample.shape != ():
				MuOutTmp = Mu[self.output] + np.dot(self.InvSigmaOutIn[i],
													(sample[self.input] - Mu[self.input]).T)

				if variance_type == 'full':
					self.SigmaOutTmp[i] = MuOutTmp ** 2 + (Sigma[np.ix_(self.output,
																		self.output)] - \
														   np.dot(self.InvSigmaOutIn[i], \
																  Sigma[np.ix_(self.input,
																			   self.output)])) ** 2

				else:
					if has_changed:
						self.SigmaOutTmp[i] = Sigma[np.ix_(self.output, self.output)] - \
											  np.dot(self.InvSigmaOutIn[i], \
													 Sigma[np.ix_(self.input, self.output)])

			else:
				MuOutTmp = Mu[self.output] + (np.dot(Sigma[np.ix_(self.output, self.input)], \
													 self.InvSigmaInIn[i]) * (
											  sample - Mu[self.input]).T).T
				if has_changed:
					self.SigmaOutTmp[i] = Sigma[np.ix_(self.output, self.output)] - \
										  np.dot(self.InvSigmaOutIn[i], \
												 Sigma[np.ix_(self.input, self.output)])

			MusOut[i, :] = MuOutTmp

			MuOut = MuOut + beta[i] * MuOutTmp

			if variance_type == 'full':
				# see course ML : nonlinearRegression.pdf
				SigmaOut = SigmaOut + beta[i] * self.SigmaOutTmp[i] - (beta[
																		   i] * MuOutTmp) ** 2

			else:
				SigmaOut = SigmaOut + beta[i] * self.SigmaOutTmp[i]

			SigmasOut[i, :, :] = self.SigmaOutTmp[i]

			# dist = np.dot(np.dot())

			for d_hist in out_hist:
				histogram[d_hist] = histogram[d_hist] + beta[i] * \
														np.exp(-(span[d_hist] - MuOutTmp[
															d_hist]) ** 2 / (2 *
																			 self.SigmaOutTmp[
																				 i][d_hist][
																				 d_hist] ** 2))

		return (MuOut, SigmaOut, histogram)

	# @jit
	# @profile
	def predict_local(self, sample, input, output, variance_type='full', reg=1e-9):
		""" Gaussian mixture regression with different input for each states

		:param sample: 	np.array((nb_states,)) or np.array((nb_input_dim, nb_states))
			Data observed in input dimension for each states
		:param input:	list i.e. : [0] or [0,1]
			List of input dimension
		:param output:list i.e. : [0] or [0,1]
			List of output dimension
		:param variance_type:
		:return:
		"""
		# set input and output
		if sample.ndim == 1:
			sample = sample[np.newaxis, :]

		has_changed = False

		if self.input != input or self.output != output:
			has_changed = True
			self.input = input
			self.output = output

			# create slices for input-output
			self.sloo = np.ix_(self.output, self.output)
			self.slii = np.ix_(self.input, self.input)
			self.sloi = np.ix_(self.output, self.input)
			self.slio = np.ix_(self.input, self.output)

		prob = np.empty(self.gmm.n_components, dtype=float)

		# 0.1 ms for get_pdf
		for i in range(self.gmm.n_components):
			prob[i] = self.get_pdf(i, sample[:, i], has_changed=has_changed, reg=reg) * self.gmm.weights_[i]

		sum_prob = np.sum(prob)
		if sum_prob:
			beta = prob / sum_prob
		else:
			beta = np.ones(prob.shape) / prob.shape[0]
		#
		# print beta
		# print "*****"

		MuOut = np.zeros(len(self.output), dtype=float)
		SigmaOut = np.zeros((len(self.output), len(self.output)), dtype=float)

		for i in range(self.gmm.n_components):
			Mu = self.gmm.means_[i]
			Sigma = self.gmm.covars_[i]

			if has_changed:
				self.InvSigmaInIn[i] = linalg.inv(Sigma[self.slii])
				self.InvSigmaOutIn[i] = np.dot(Sigma[self.sloi],
											   self.InvSigmaInIn[i])


			MuOutTmp = Mu[self.output] + np.dot(self.InvSigmaOutIn[i],
												(sample[:, i] - Mu[self.input]).T)


			if variance_type == 'full':
				self.SigmaOutTmp[i] = MuOutTmp ** 2 + (Sigma[self.sloo] - \
													   np.dot(self.InvSigmaOutIn[i], \
															  Sigma[self.slio])) ** 2
			else:
				if has_changed:
					self.SigmaOutTmp[i] = Sigma[self.sloo] - \
										  np.dot(self.InvSigmaOutIn[i], \
												 Sigma[self.slio])

			MuOut = MuOut + beta[i] * MuOutTmp

			if variance_type == 'full':
				# see course ML : nonlinearRegression.pdf
				SigmaOut = SigmaOut + beta[i] * self.SigmaOutTmp[i] - (beta[
																		   i] * MuOutTmp) ** 2

			else:
				SigmaOut = SigmaOut + beta[i] * self.SigmaOutTmp[i]

		return (MuOut, SigmaOut)

	def predict(self, sample, input, output, variance_type='full', sigma_input=None):

		# X = np.asarray(X, dtype=np.float)
		"""
		if X.ndim == 1:
			X = X[:, np.newaxis]
		if X.shape[0] < self.gmm.n_components:
			raise ValueError(
				'GMM estimation with %s components, but got only %s samples' %
				(self.gmm.n_components, X.shape[0]))
		"""
		# set input and output

		has_changed = False

		if self.input != input or self.output != output or sigma_input is not None:
			has_changed = True
			self.input = input
			self.output = output

		inpCr = np.array(self.input, dtype=np.intp)
		outCr = np.array(self.output, dtype=np.intp)

		prob = np.empty(self.gmm.n_components, dtype=float)

		# 0.1 ms for get_pdf
		for i in range(self.gmm.n_components):
			prob[i] = self.get_pdf(i, sample, has_changed=has_changed) * self.gmm.weights_[
				i]

		sum_prob = np.sum(prob)
		if sum_prob:
			beta = prob / sum_prob
		else:
			beta = np.ones(prob.shape) / prob.shape[0]

		sloo = np.ix_(self.output, self.output)
		slii = np.ix_(self.input, self.input)
		sloi = np.ix_(self.output, self.input)
		slio = np.ix_(self.input, self.output)

		MuOut = np.zeros(len(self.output), dtype=float)
		SigmaOut = np.zeros((len(self.output), len(self.output)), dtype=float)

		for i in range(self.gmm.n_components):
			Mu = self.gmm.means_[i]
			Sigma = self.gmm.covars_[i]

			if has_changed:
				if sigma_input is None:
					self.InvSigmaInIn[i] = linalg.inv(Sigma[slii] + 1e-5 * np.eye(len(self.input)))
				else:
					self.InvSigmaInIn[i] = linalg.inv(Sigma[slii] + sigma_input)

				self.InvSigmaOutIn[i] = np.dot(Sigma[sloi],
											   self.InvSigmaInIn[i])

			"""
			MuOutTmp = Mu[self.output] + np.dot((Sigma[np.ix_(self.output, self.input)] * InvSigmaInIn) * \
										 (sample[0,self.input] - Mu[self.input]).T).T
			"""
			if sample.shape != ():
				MuOutTmp = Mu[self.output] + np.dot(self.InvSigmaOutIn[i],
													(sample - Mu[self.input]).T)

				if variance_type == 'full':
					self.SigmaOutTmp[i] = MuOutTmp ** 2 + (Sigma[sloo] - \
														   np.dot(self.InvSigmaOutIn[i], \
																  Sigma[slio])) ** 2

				else:
					if has_changed:
						self.SigmaOutTmp[i] = Sigma[sloo] - \
											  np.dot(self.InvSigmaOutIn[i], \
													 Sigma[slio])

			else:
				MuOutTmp = Mu[self.output] + (np.dot(Sigma[sloi], \
													 self.InvSigmaInIn[i]) * (
											  sample - Mu[self.input]).T).T
				if has_changed:
					self.SigmaOutTmp[i] = Sigma[sloo] - \
										  np.dot(self.InvSigmaOutIn[i], \
												 Sigma[slio])

			MuOut = MuOut + beta[i] * MuOutTmp

			if variance_type == 'full':
				# see course ML : nonlinearRegression.pdf
				SigmaOut = SigmaOut + beta[i] * self.SigmaOutTmp[i] - (beta[
																		   i] * MuOutTmp) ** 2

			else:
				SigmaOut = SigmaOut + beta[i] * self.SigmaOutTmp[i]

		return (MuOut, SigmaOut)

	def get_pdf(self, i, sample, has_changed=True, reg=1e-9):
		if sample.shape != ():
			D_tmp = sample - self.gmm.means_[i][self.input]
		else:
			D_tmp = sample - self.gmm.means_[i][self.input]

		if has_changed:
			slii = np.ix_(self.input, self.input)
			# self.cov_tmp[i] = np.empty((len(self.input), len(self.input)), dtype=float)
			# k = 0;
			# for i_ in self.input:
			# 	l = 0;
			# 	for j_ in self.input:
			# 		self.cov_tmp[i][k][l] = self.gmm.covars_[i][i_][j_]
			# 		l += 1
			# 	k += 1
			#
			self.cov_tmp[i] = self.gmm.covars_[i][slii] + reg * np.eye(len(self.input))
			self.inv_tmp[i] = linalg.inv(self.cov_tmp[i])
			self.pri_tmp[i] = np.sqrt(pow(2 * np.pi, len(self.input)) * \
									  (np.abs(linalg.det(self.cov_tmp[i]))))

		dist = np.dot(np.dot(D_tmp, self.inv_tmp[i]), D_tmp)
		prob = np.exp(-0.5 * dist) / self.pri_tmp[i]

		return prob

	def get_pdf_un(self, i, sample, has_changed=True):
		if sample.shape != ():
			D_tmp = sample - self.gmm.means_[i][self.input]
		else:
			D_tmp = sample - self.gmm.means_[i][self.input]

		if has_changed:
			slii = np.ix_(self.input, self.input)
			# self.cov_tmp[i] = np.empty((len(self.input), len(self.input)), dtype=float)
			# k = 0;
			# for i_ in self.input:
			# 	l = 0;
			# 	for j_ in self.input:
			# 		self.cov_tmp[i][k][l] = self.gmm.covars_[i][i_][j_]
			# 		l += 1
			# 	k += 1
			#
			self.cov_tmp[i] = self.gmm.covars_[i][slii]
			self.inv_tmp[i] = linalg.inv(self.cov_tmp[i])
			self.pri_tmp[i] = np.sqrt(pow(2 * np.pi, len(self.input)) * \
									  (np.abs(linalg.det(self.cov_tmp[i]))))

		dist = np.dot(np.dot(D_tmp, self.inv_tmp[i]), D_tmp)
		prob = np.exp(-0.5 * dist) # / self.pri_tmp[i]

		return prob