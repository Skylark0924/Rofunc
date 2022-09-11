import numpy as np


def gaussian_moment_matching(mus, sigmas, h=None):
	"""

	:param mus:			[np.array([nb_states, nb_timestep, nb_dim])]
				or [np.array([nb_states, nb_dim])]
	:param sigmas:		[np.array([nb_states, nb_timestep, nb_dim, nb_dim])]
				or [np.array([nb_states, nb_dim, nb_dim])]
	:param h: 			[np.array([nb_timestep, nb_states])]
	:return:
	"""

	if h is None:
		h = np.ones((mus.shape[1], mus.shape[0]))/ mus.shape[0]

	if h.ndim == 1:
		h = h[None]

	if mus.ndim == 3:
		mu = np.einsum('ak,kai->ai', h, mus)
		dmus = mus - mu[None]  # nb_timesteps, nb_states, nb_dim
		if sigmas.ndim == 4:
			sigma = np.einsum('ak,kaij->aij', h, sigmas) + \
				 np.einsum('ak,akij->aij', h, np.einsum('kai,kaj->akij', dmus, dmus))
		else:
			sigma = np.einsum('ak,kij->aij', h, sigmas) + \
				 np.einsum('ak,akij->aij', h, np.einsum('kai,kaj->akij', dmus, dmus))

		return mu, sigma
	else:
		mu = np.einsum('ak,ki->ai', h, mus)
		dmus = mus[None] - mu[:, None] # nb_timesteps, nb_states, nb_dim
		sigma = np.einsum('ak,kij->aij', h, sigmas) + \
				 np.einsum('ak,akij->aij',h , np.einsum('aki,akj->akij', dmus, dmus))

		return mu, sigma


def gaussian_conditioning(mu, sigma, data_in, dim_in, dim_out, reg=None):
	"""

	:param mu: 			[np.array([nb_timestep, nb_dim])]
	:param sigma: 		[np.array([nb_timestep, nb_dim, nb_dim])]
	:param data_in: 	[np.array([nb_timestep, nb_dim])]
	:param dim_in: 		[slice]
	:param dim_out: 	[slice]
	:return:
	"""
	if sigma.ndim == 2:

		if reg is None:
			inv_sigma_in_in = np.linalg.inv(sigma[dim_in, dim_in])
		else:
			reg = reg * np.eye(dim_in.stop - dim_in.start)
			inv_sigma_in_in = np.linalg.inv(sigma[dim_in, dim_in] + reg)

		inv_sigma_out_in = np.einsum('ji,jk->ik', sigma[dim_in, dim_out], inv_sigma_in_in)
		mu_cond = mu[dim_out] + np.einsum('ij,aj->ai', inv_sigma_out_in,
											 data_in - mu[dim_in])
		sigma_cond = sigma[dim_out, dim_out] - np.einsum('ij,jk->ik', inv_sigma_out_in,
															sigma[dim_in, dim_out])

	else:

		if reg is None:
			inv_sigma_in_in = np.linalg.inv(sigma[:, dim_in, dim_in])
		else:
			reg = reg * np.eye(dim_in.stop - dim_in.start)
			inv_sigma_in_in = np.linalg.inv(sigma[:, dim_in, dim_in] + reg)

		inv_sigma_out_in = np.einsum('aji,ajk->aik', sigma[:, dim_in, dim_out], inv_sigma_in_in)
		mu_cond = mu[:, dim_out] + np.einsum('aij,aj->ai', inv_sigma_out_in,
											 data_in - mu[:, dim_in])
		sigma_cond = sigma[:, dim_out, dim_out] - np.einsum('aij,ajk->aik', inv_sigma_out_in,
															sigma[:, dim_in, dim_out])

	return mu_cond, sigma_cond


def renyi_entropy(p):
	"""
	Finds renyi entropy of Gaussian
	:param p: pbdlib.gmm.GMM
	"""

	mu = np.transpose(p.mu, axes=(1, 0, 2))
	pi = p.priors

	sigma_ = np.transpose(p.sigma, axes=(1, 0, 2, 3))
	if sigma_.ndim != 4:
		sigma = np.tile(sigma_[None], (np.shape(mu)[0], 1, 1, 1))
	else:
		sigma = sigma_

	sigma_inv = np.linalg.inv(sigma)

	K = mu.shape[1]
	D = mu.shape[-1]

	P = np.einsum('na,nb->nab', pi, pi)

	sigma_inv_tiled = np.tile(sigma_inv[:, None], (1, K, 1, 1, 1))
	sigma_inv_tiled_T = np.transpose(sigma_inv_tiled, axes=(0, 2, 1, 3, 4))
	sigma_inv_sum = sigma_inv_tiled + sigma_inv_tiled_T
	sigma_inv_sum_inv = np.linalg.inv(sigma_inv_sum)  # (N,K,K,dim,dim)

	mult_op2 = np.einsum('naij,naj->nai', sigma_inv, mu)
	mult_op2_tiled = np.tile(mult_op2[:, None], (1, K, 1, 1))
	mult_op2_tiled_T = np.transpose(mult_op2_tiled, axes=(0, 2, 1, 3))
	mult_op2_sum = mult_op2_tiled + mult_op2_tiled_T  # (N,K,K,dim)

	sum_op2 = np.einsum('nai,nai->na', mu, mult_op2)
	sum_op2_tiled = np.tile(sum_op2[:, None], (1, K, 1))
	sum_op2_tiled_T = np.transpose(sum_op2_tiled, axes=(0, 2, 1))
	sum_op2_sum = sum_op2_tiled + sum_op2_tiled_T  # (N,K,K)

	muij = np.einsum('nabij, nabj->nabi', sigma_inv_sum_inv, mult_op2_sum)  # (N,K,K,dim)

	sum_op1 = np.einsum('nabi,nabi->nab', muij, np.einsum('nabij,nabj->nabi', sigma_inv_sum, muij))  # (N,K,K)

	delta_1 = (sum_op1 - sum_op2_sum)  # (N,K,K)
	sigma_inv_sum_det = - np.linalg.slogdet(sigma_inv_sum)[1]  # (N,K,K)
	sigma_inv_det = np.linalg.slogdet(sigma_inv)[1]  # (N,K)

	sigma_inv_det_tiled = np.tile(sigma_inv_det[:, None], (1, K, 1))
	sigma_inv_det_tiled_T = np.transpose(sigma_inv_det_tiled, axes=(0, 2, 1))
	sigma_inv_det_sum = sigma_inv_det_tiled + sigma_inv_det_tiled_T  # (N,K,K)

	sum_op3 = sigma_inv_sum_det + sigma_inv_det_sum  # (N,K,K)

	sum_op4 = -np.ones_like(delta_1) * D * np.float32(np.log(2 * np.pi))

	delta = np.exp(0.5 * (delta_1 + sum_op3 + sum_op4))
	res = np.sum(np.einsum('nab,nab->nab', P, delta), (1, 2))  # (N)

	return -np.log(res)
