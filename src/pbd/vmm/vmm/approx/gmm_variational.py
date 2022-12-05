from .distributions import GMMDiag, GMMFull, MoE
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from ..utils.tf_utils import log_normalize

class GMMApprox(object):
	def __init__(self, log_unnormalized_prob, gmm=None, k=10, loc=0., std=1., ndim=None, loc_tril=None,
				 samples=20, temp=1., cov_type='diag'):
		"""

		:param log_unnormalized_prob:	Unnormalized log density to estimate
		:type log_unnormalized_prob: 	a tensorflow function that takes [batch_size, ndim]
			as input and returns [batch_size]
		:param gmm:
		:param k:		number of components for GMM approximation
		:param loc:		for initialization, mean
		:param std:		for initialization, standard deviation
		:param ndim:
		"""
		self.log_prob = log_unnormalized_prob
		self.ndim = ndim
		self.temp = temp

		if gmm is None:
			assert ndim is not None, "If no gmm is defined, should give the shape of x"

			if cov_type == 'diag':
				log_priors = tf.Variable(10. * tf.ones(k))
				locs = tf.Variable(tf.random_normal((k, ndim), loc, std))
				log_std_diags = tf.Variable(tf1.log(std/k * tf.ones((k, ndim))))

				gmm = GMMDiag(log_priors=log_priors,
							  locs=locs,
							  log_std_diags=log_std_diags)
			elif cov_type == 'full':

				log_priors = tf.Variable(10. * tf.ones(k))
				locs = tf.Variable(tf1.random_normal((k, ndim), loc, std))
				loc_tril = loc_tril if loc_tril is not None else std/k
				tril_cov = tf.Variable(loc_tril ** 2 * tf.eye(ndim, batch_shape=(k, )))

				gmm = GMMFull(log_priors=log_priors,
							  locs=locs,
							  tril_cov=tril_cov)
			else:
				raise ValueError("Unrecognized covariance type")



		self.num_samples = samples
		self.gmm = gmm

	@property
	def sample_shape(self):
		return (self.num_samples, )

	@property
	def opt_params(self):
		"""
		Parameters to train
		:return:
		"""
		return self.gmm.opt_params

	def mixture_lower_bound(self, k):
		samples = self.gmm.component_sample(k, self.sample_shape)
		log_qs = self.gmm.log_prob(samples)
		log_ps = self.temp * self.log_prob(samples)

		return tf.reduce_mean(log_ps - log_qs)

	def mixture_elbo_fast(self, *args):
		samples_conc = tf.reshape(
			tf.transpose(self.gmm.all_components_sample(self.sample_shape), perm=(1, 0, 2))
		, (-1, self.ndim)) # [k * nsamples, ndim]

		log_qs = tf.reshape(self.gmm.log_prob(samples_conc), (self.gmm.k, self.num_samples))
		log_ps = tf.reshape(self.temp * self.log_prob(samples_conc), (self.gmm.k, self.num_samples))

		component_elbos = tf.reduce_mean(log_ps-log_qs, axis=1)

		return tf.reduce_sum(component_elbos * tf.exp(log_normalize(self.gmm.log_priors)))
		# log_qs =

	def mixture_elbo(self):
		component_elbos = tf.stack([self.mixture_lower_bound(k)
									for k in range(self.gmm.k)])

		return tf.reduce_sum(component_elbos * tf.exp(log_normalize(self.gmm.log_priors)))

	@property
	def cost(self):
		return -self.mixture_elbo_fast()
		# return -self.mixture_elbow()

class GMMApproxCond(GMMApprox):
	def __init__(self, log_unnormalized_prob, moe=None, ndim_in=None, ndim_out=None, **kwargs):
		"""

		:param log_unnormalized_prob:	Unnormalized log density of the conditional model to estimate log p(y | x)
		:type log_unnormalized_prob: 	a tensorflow function that takes [batch_size, ndim_y] [batch_size, ndim_x]
			as input and returns [batch_size]
		:param moe:
		:type moe: MoE
		"""
		self.ndim_in = ndim_in  # x
		self.ndim_out = ndim_out  # y

		GMMApprox.__init__(self, log_unnormalized_prob, gmm=moe, **kwargs)

	@property
	def moe(self):
		return self.gmm

	def mixture_elbow_fast(self, x):
		# importance sample mixture weights
		samples, weights = self.moe.sample_is(x)

		loq_qs = self.moe.log_prob(samples, x)
		log_ps = self.temp * self.log_prob(y, x)

		elbos_is = tf.reduce_mean(log_ps-loq_qs, axis=1)

		return tf.reduce_sum(elbos_is * tf1.exp(weights))

	def cost(self, x):
		return -self.mixture_elbow_fast(x)