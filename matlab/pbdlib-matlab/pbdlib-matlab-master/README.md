# pbdlib-matlab

<p>PbDlib is a collection of source codes for robot programming by demonstration (learning from demonstration). It includes various functionalities including statistical learning, dynamical systems, optimal control and Riemannian geometry.</p>
<p>PbDlib can be used in applications requiring task adaptation, human-robot skill transfer, safe controllers based on minimal intervention principle, as well as for probabilistic motion analysis and synthesis in multiple coordinate systems.</p>
<p>The codes are compatible with both Matlab and GNU Octave. Other versions of the library in Python and C++ are also available at https://www.idiap.ch/software/pbdlib/ (currently, the Matlab version contains the most complete set of examples).</p>

### Usage

Examples starting with `demo_` can be run as examples. The corresponding publications related to these examples are listed below. 

### List of examples

All the examples are located in the `demos` folder, and the functions are located in the `m_fcts` folder.

| Filename | Ref. | Description |
|----------|------|-------------|
| [demo_affineTransform01.m](./demos/demo_affineTransform01.m) | [[1]](#ref-1) | Miscellaneous affine transformations of raw data as pre-processing step to train a task-parameterized model |
| [demo_AR01.m](./demos/demo_AR01.m) | [[2]](#ref-2) | Multivariate autoregressive (AR) model parameters estimation with least-squares |
| [demo_AR_HSMM01.m](./demos/demo_AR_HSMM01.m) | [[2]](#ref-2) | Multivariate autoregressive (AR) model implemented as a hidden semi-Markov model with lognormal duration model |
| [demo_Bezier01.m](./demos/demo_Bezier01.m) | [[5]](#ref-5) | Bezier curves as a superposition of Bernstein polynomials |
| [demo_Bezier02.m](./demos/demo_Bezier02.m) | [[5]](#ref-5) | Bezier curves fitting |
| [demo_Bezier_illustr01.m](./demos/demo_Bezier_illustr01.m) | [[5]](#ref-5) | Fitting Bezier curves of different degrees |
| [demo_Bezier_illustr02.m](./demos/demo_Bezier_illustr02.m) | [[5]](#ref-5) | Illustration of linear, quadratic and cubic Bezier curves |
| [demo_covariance01.m](./demos/demo_covariance01.m) | [[10]](#ref-10) | Covariance computation in matrix form |
| [demo_DMP01.m](./demos/demo_DMP01.m) | [[2]](#ref-2) | Dynamic movement primitive (DMP) encoding with radial basis functions |
| [demo_DMP02.m](./demos/demo_DMP02.m) | [[2]](#ref-2) | Generalization of dynamic movement primitive (DMP) with polynomial fitting using radial basis functions |
| [demo_DMP_batchLQR01.m](./demos/demo_DMP_batchLQR01.m) | [[2]](#ref-2) | Emulation of DMP with a spring system controlled by batch LQR |
| [demo_DMP_GMR01.m](./demos/demo_DMP_GMR01.m) | [[2]](#ref-2) | Emulation of DMP by using a GMM with diagonal covariance matrix, and retrieval computed with GMR |
| [demo_DMP_GMR02.m](./demos/demo_DMP_GMR02.m) | [[2]](#ref-2) | Same as demo_DMP_GMR01.m with full covariance |
| [demo_DMP_GMR03.m](./demos/demo_DMP_GMR03.m) | [[2]](#ref-2) | Same as demo_DMP_GMR02.m with GMR used to regenerate the path of a spring-damper system, resulting in a nonlinear force profile |
| [demo_DMP_GMR04.m](./demos/demo_DMP_GMR04.m) | [[2]](#ref-2) | Same as demo_DMP_GMR03.m by using the task-parameterized model formalism |
| [demo_DMP_GMR_illustr01.m](./demos/demo_DMP_GMR_illustr01.m) | [[2]](#ref-2) | Illustration of DMP with GMR to regenerate the nonlinear force profile | 
| [demo_DMP_GMR_LQR01.m](./demos/demo_DMP_GMR_LQR01.m) | [[2]](#ref-2) | Same example as demo_DMP_GMR04.m, but with LQR |
| [demo_DPmeans_online01.m](./demos/demo_DPmeans_online01.m) | [[7]](#ref-7) | Online clustering with DP-Means algorithm |
| [demo_DPmeans_online02.m](./demos/demo_DPmeans_online02.m) | [[7]](#ref-7) | Online clustering with DP-Means algorithm, with stochastic samples |
| [demo_DTW01.m](./demos/demo_DTW01.m) | [[10]](#ref-10) | Trajectory realignment through dynamic time warping (DTW) |
| [demo_ergodicControl_1D01.m](./demos/demo_ergodicControl_1D01.m) | [[5]](#ref-5) | 1D ergodic control with spectral multiscale coverage (SMC) algorithm |
| [demo_ergodicControl_2D01.m](./demos/demo_ergodicControl_2D01.m) | [[5]](#ref-5) | 2D ergodic control with spectral multiscale coverage (SMC) algorithm |
| [demo_ergodicControl_3D01.m](./demos/demo_ergodicControl_3D01.m) | [[5]](#ref-5) | 3D ergodic control with spectral multiscale coverage (SMC) algorithm |
| [demo_ergodicControl_nD01.m](./demos/demo_ergodicControl_nD01.m) | [[5]](#ref-5) | nD ergodic control with spectral multiscale coverage (SMC) algorithm |
| [demo_Gaussian01.m](./demos/demo_Gaussian01.m) | [[5]](#ref-5) | Use of Chi-square values to determine the percentage of data within the contour of a multivariate normal distribution |
| [demo_Gaussian02.m](./demos/demo_Gaussian02.m) | [[5]](#ref-5) | Conditional probability with a multivariate normal distribution |
| [demo_Gaussian03.m](./demos/demo_Gaussian03.m) | [[5]](#ref-5) | Gaussian conditioning with uncertain inputs |
| [demo_Gaussian04.m](./demos/demo_Gaussian04.m) | [[5]](#ref-5) | Gaussian estimate of a GMM with the law of total covariance |
| [demo_Gaussian05.m](./demos/demo_Gaussian05.m) | [[5]](#ref-5) | Stochastic sampling with multivariate Gaussian distribution |
| [demo_Gaussian06.m](./demos/demo_Gaussian06.m) | [[5]](#ref-5) | Gaussian reformulated as zero-centered Gaussian with augmented covariance |
| [demo_Gaussian_illustr01.m](./demos/demo_Gaussian_illustr01.m) | [[5]](#ref-5) | Illustration of Gaussian conditioning with uncertain inputs |
| [demo_GaussProd4nullspace_2D01.m](./demos/demo_GaussProd4nullspace_2D01.m) | [[10]](#ref-10) | 2D illustration of using a product of Gaussians to compute the hierarchy of two tasks |
| [demo_GaussProd4nullspace_3D01.m](./demos/demo_GaussProd4nullspace_3D01.m) | [[10]](#ref-10) | 3D illustration of using a product of Gaussians to compute the hierarchy of three tasks |
| [demo_GaussProd_interp_illustr01.m](./demos/demo_GaussProd_interp_illustr01.m) | [[10]](#ref-10) | Smooth transition between hierarchy constraints by relying on SPD geodesics |
| [demo_GMM01.m](./demos/demo_GMM01.m) | [[5]](#ref-5) | Gaussian mixture model (GMM) parameters estimation |
| [demo_GMM02.m](./demos/demo_GMM02.m) | [[5]](#ref-5) | GMM with different covariance structures |
| [demo_GMM_augmSigma01.m](./demos/demo_GMM_augmSigma01.m) | [[5]](#ref-5) | Gaussian mixture model (GMM) parameters estimation with zero means and augmented covariances |
| [demo_GMM_EM01.m](./demos/demo_GMM_EM01.m) | [[5]](#ref-5) | Problem of local optima in EM for GMM parameters estimation |
| [demo_GMM_HDDC01.m](./demos/demo_GMM_HDDC01.m) | [[5]](#ref-5) | High Dimensional Data Clustering (HDDC, or HD-GMM) model from Bouveyron (2007) |
| [demo_GMM_logGMM01.m](./demos/demo_GMM_logGMM01.m) | [[5]](#ref-5) | Multivariate lognormal mixture model parameters estimation with EM algorithm |
| [demo_GMM_logNormal01.m](./demos/demo_GMM_logNormal01.m) | [[5]](#ref-5) | Conditional probability with multivariate lognormal distribution |
| [demo_GMM_MFA01.m](./demos/demo_GMM_MFA01.m) | [[5]](#ref-5) | Mixture of factor analyzers (MFA) |
| [demo_GMM_MPPCA01.m](./demos/demo_GMM_MPPCA01.m) | [[5]](#ref-5) | Mixture of probabilistic principal component analyzers (MPPCA) |
| [demo_GMM_profileGMM01.m](./demos/demo_GMM_profileGMM01.m) | [[5]](#ref-5) | Univariate velocity profile fitting with a GMM and a weighted EM algorithm |
| [demo_GMM_profileGMM_multivariate01.m](./demos/demo_GMM_profileGMM_multivariate01.m) | [[5]](#ref-5) | Multivariate velocity profile fitting with a GMM and a weighted EM algorithm |
| [demo_GMM_profileLogGMM01.m](./demos/demo_GMM_profileLogGMM01.m) | [[5]](#ref-5) | Univariate velocity profile fitting with a lognormal mixture model (GMM) and a weighted EM algorithm |
| [demo_GMM_profileLogGMM_multivariate01.m](./demos/demo_GMM_profileLogGMM_multivariate01.m) | [[5]](#ref-5) | Multivariate velocity profile fitting with a GMM and a weighted EM algorithm |
| [demo_GMM_semiTied01.m](./demos/demo_GMM_semiTied01.m) | [[5]](#ref-5) | Semi-tied Gaussian Mixture Model by tying the covariance matrices of a GMM with a set of common basis vectors |
| [demo_GMR01.m](./demos/demo_GMR01.m) | [[5]](#ref-5) | Gaussian mixture model (GMM) and time-based Gaussian mixture regression (GMR) used for reproduction |
| [demo_GMR02.m](./demos/demo_GMR02.m) | [[5]](#ref-5) | GMR computed with precision matrices instead of covariances |
| [demo_GMR03.m](./demos/demo_GMR03.m) | [[5]](#ref-5) | Chain rule with Gaussian conditioning |
| [demo_GMR_3Dviz01.m](./demos/demo_GMR_3Dviz01.m) | [[5]](#ref-5) | 3D visualization of a Gaussian mixture model (GMM) with time-based Gaussian mixture regression (GMR) used for reproduction |
| [demo_GMR_augmSigma01.m](./demos/demo_GMR_augmSigma01.m) | [[5]](#ref-5) | GMR with Gaussians reparamterized to have zero means and augmented covariances |
| [demo_GMR_DS01.m](./demos/demo_GMR_DS01.m) | [[2]](#ref-2) | Gaussian mixture model with GMR and dynamical systems used for reproduction, with decay variable used as input (as in DMP) |
| [demo_GMR_polyFit01.m](./demos/demo_GMR_polyFit01.m) | [[5]](#ref-5) | Polynomial fitting with multivariate GMR |
| [demo_GMR_probTraj01.m](./demos/demo_GMR_probTraj01.m) | [[5]](#ref-5) | Probabilistic trajectory generation with GMR obtained from normally distributed GMM centers |
| [demo_GMR_SEDS01.m](./demos/demo_GMR_SEDS01.m) | [[2]](#ref-2) | Continuous autonomous dynamical system, with GMR using a constrained optimization similar to the SEDS approach |
| [demo_GMR_SEDS_augmSigma01.m](./demos/demo_GMR_SEDS_augmSigma01.m) | [[2]](#ref-2) | Same as demo_GMR_SEDS01.m with an augmented state-space encoding |
| [demo_GMR_SEDS_discrete01.m](./demos/demo_GMR_SEDS_discrete01.m) | [[2]](#ref-2) | Same as demo_GMR_SEDS01.m but with discrete autonomous dynamical system |
| [demo_GMR_SEDS_discrete_augmSigma01.m](./demos/demo_GMR_SEDS_discrete_augmSigma01.m) | [[2]](#ref-2) | Same as demo_GMR_SEDS_augmSigma01.m but with discrete autonomous dynamical system |
| [demo_GMR_wrapped01.m](./demos/demo_GMR_wrapped01.m) | [[5]](#ref-5) | Wrapped GMM and wrapped GMR in 2D (with only the first dimension being periodic) |
| [demo_GPR01.m](./demos/demo_GPR01.m) | [[2]](#ref-2) | Gaussian process regression (GPR) |
| [demo_GPR02.m](./demos/demo_GPR02.m) | [[2]](#ref-2) | GPR with stochastic samples from the prior and the posterior |
| [demo_GPR03.m](./demos/demo_GPR03.m) | [[2]](#ref-2) | GPR with periodic kernel function |
| [demo_GPR04.m](./demos/demo_GPR04.m) | [[2]](#ref-2) | GPR with Matern kernel function |
| [demo_GPR05.m](./demos/demo_GPR05.m) | [[2]](#ref-2) | GPR for motion generation with new targets |
| [demo_GPR_anim01.m](./demos/demo_GPR_anim01.m) | [[2]](#ref-2) | Gaussian process regression (GPR) with prior distribution formed so that it can be smoothly animated for illustration purpose |
| [demo_GPR_closedShape01.m](./demos/demo_GPR_closedShape01.m) | [[2]](#ref-2) | Closed shape modeling with Gaussian process regression (GPR) |
| [demo_GPR_closedShape02.m](./demos/demo_GPR_closedShape02.m) | [[2]](#ref-2) | Gaussian process implicit surface (GPIS) representation with thin-plate covariance function in GPR |
| [demo_GPR_GMR_illustr01.m](./demos/demo_GPR_GMR_illustr01.m) | [[2]](#ref-2) | Illustration of the different notions of variance for GPR and GMR |
| [demo_GPR_linTrend01.m](./demos/demo_GPR_linTrend01.m) | [[2]](#ref-2) | Gaussian process regression (GPR) with linear trend |
| [demo_GPR_paramOptim01.m](./demos/demo_GPR_paramOptim01.m) | [[2]](#ref-2) | GPR with optimization of the kernel parameters |
| [demo_GPR_recursive01.m](./demos/demo_GPR_recursive01.m) | [[2]](#ref-2) | Recursive computation of Gaussian process regression (GPR) |
| [demo_GPR_TP01.m](./demos/demo_GPR_TP01.m) | [[1]](#ref-1) | Use of GPR as a task-parameterized model, with DS-GMR used to retrieve continuous movements |
| [demo_grabData01.m](./demos/demo_grabData01.m) | [[1]](#ref-1) | Collect movement data from mouse cursor |
| [demo_gradientDescent01.m](./demos/demo_gradientDescent01.m) | [[10]](#ref-10) | Optimization with gradient descent for 1D input (Newton's method) |
| [demo_gradientDescent02.m](./demos/demo_gradientDescent02.m) | [[10]](#ref-10) | Optimization with gradient descent for 2D input (Newton's method) |
| [demo_gradientDescent03.m](./demos/demo_gradientDescent03.m) | [[10]](#ref-10) | Optimization with gradient descent for 1D input and 2D output (Gauss-Newton algorithm) |
| [demo_gradientDescent04.m](./demos/demo_gradientDescent04.m) | [[10]](#ref-10) | Optimization with gradient descent for 2D input and 2D output depicting a planar robot IK problem (Gauss-Newton algorithm) |
| [demo_HMM01.m](./demos/demo_HMM01.m) | [[2]](#ref-2) | Hidden Markov model (HMM) with single Gaussian as emission distribution |
| [demo_HMM02.m](./demos/demo_HMM02.m) | [[2]](#ref-2) | Emulation of HSMM with a standard HMM |
| [demo_HMM_Viterbi01.m](./demos/demo_HMM_Viterbi01.m) | [[2]](#ref-2) | Viterbi decoding in HMM to estimate best state sequence from observations |
| [demo_HSMM01.m](./demos/demo_HSMM01.m) | [[2]](#ref-2) | Variable duration model implemented as a hidden semi-Markov model (HSMM), by encoding the state duration after EM |
| [demo_HSMM02.m](./demos/demo_HSMM02.m) | [[2]](#ref-2) | Same as demo_HSMM01.m but with a log-normal duration model |
| [demo_HSMM_adaptiveDuration01.m](./demos/demo_HSMM_adaptiveDuration01.m) | [[9]](#ref-9) | Hidden semi-Markov model with adaptive duration |
| [demo_HSMM_adaptiveDuration_infHor01.m](./demos/demo_HSMM_adaptiveDuration_infHor01.m) | [[9]](#ref-9) | Same as demo_HSMM_adaptiveDuration01.m with LQR |
| [demo_HSMM_MPC01.m](./demos/demo_HSMM_MPC01.m) | [[2]](#ref-2) | Use of HSMM (with lognormal duration model) and batch LQR (with position only) for motion synthesis |
| [demo_HSMM_online01.m](./demos/demo_HSMM_online01.m) | [[2]](#ref-2) | Online HSMM |
| [demo_ID01.m](./demos/demo_ID01.m) | [[10]](#ref-10) | Inverse dynamics with dynamically consistent nullspace (requires the robotics toolbox) |
| [demo_IK01.m](./demos/demo_IK01.m) | [[10]](#ref-10) | Basic forward and inverse kinematics for a planar robot, with numerical Jacobian computation |
| [demo_IK02.m](./demos/demo_IK02.m) | [[10]](#ref-10) | Inverse kinematics with nullspace projection operator |
| [demo_IK_expForm01.m](./demos/demo_IK_expForm01.m) | [[10]](#ref-10) | Basic forward and inverse kinematics for a planar robot, with analytical Jacobian computation |
| [demo_IK_minimal01.m](./demos/demo_IK_minimal01.m) | [[10]](#ref-10) | Basic forward and inverse kinematics for a planar robot, with minimal computation |
| [demo_IK_nullspaceAsProduct01.m](./demos/demo_IK_nullspaceAsProduct01.m) | [[1]](#ref-1) | 3-level nullspace control formulated as product of Gaussians |
| [demo_IK_nullspace_TPGMM01.m](./demos/demo_IK_nullspace_TPGMM01.m) | [[1]](#ref-1) | IK with nullspace treated with task-parameterized GMM (bimanual tracking task, version with 4 frames) |
| [demo_IK_pointing_TPGMM01.m](./demos/demo_IK_pointing_TPGMM01.m) | [[1]](#ref-1) | Task-parameterized GMM to encode pointing direction by considering inverse kinematics |
| [demo_IK_quat01.m](./demos/demo_IK_quat01.m) | [[10]](#ref-10) | Inverse kinematics for orientation data with quaternion references (version with quaternion Jacobian) |
| [demo_IK_quat02.m](./demos/demo_IK_quat02.m) | [[10]](#ref-10) | Inverse kinematics for orientation data with quaternion references (version with standard Jacobian) |
| [demo_IK_weighted01.m](./demos/demo_IK_weighted01.m) | [[10]](#ref-10) | Inverse kinematics with nullspace control, by considering weights in joint space and in task space |
| [demo_ILC01.m](./demos/demo_ILC01.m) | [[10]](#ref-10) | Iterative correction of errors for a recurring movement with ILC |
| [demo_IPRA01.m](./demos/demo_IPRA01.m) | [[10]](#ref-10) | Gaussian mixture model (GMM) learned with iterative pairwise replacement algorithm (IPRA) |
| [demo_Kalman01.m](./demos/demo_Kalman01.m) | [[10]](#10) | Kalman filter computed as a feedback term or as a product of Gaussians |
| [demo_kernelPCA01.m](./demos/demo_kernelPCA01.m) | [[10]](#ref-10) | Kernel PCA, with comparison to PCA |
| [demo_LS01.m](./demos/demo_LS01.m) | [[10]](#ref-10) | Multivariate ordinary least squares |
| [demo_LS_IRLS01.m](./demos/demo_LS_IRLS01.m) | [[10]](#ref-10) | Iteratively reweighted least squares |
| [demo_LS_IRLS_logisticRegression01.m](./demos/demo_LS_IRLS_logisticRegression01.m) | [[10]](#ref-10) | Logistic regression computed with iteratively reweighted least squares (IRLS) algorithm |
| [demo_LS_IRLS_logisticRegression02.m](./demos/demo_LS_IRLS_logisticRegression02.m) | [[10]](#ref-10) | Logistic regression with multivariate inputs computed with IRLS algorithm |
| [demo_LS_polFit01.m](./demos/demo_LS_polFit01.m) | [[10]](#ref-10) | Polynomial fitting with least squares |
| [demo_LS_recursive01.m](./demos/demo_LS_recursive01.m) | [[10]](#ref-10) | Recursive computation of least squares estimate (implementation with block data) |
| [demo_LS_recursive02.m](./demos/demo_LS_recursive02.m) | [[10]](#ref-10) | Recursive computation of least squares estimate with one datapoint at a time |
| [demo_LS_weighted01.m](./demos/demo_LS_weighted01.m) | [[10]](#ref-10) | Weighted least squares regression |
| [demo_LWR01.m](./demos/demo_LWR01.m) | [[5]](#ref-5) | Locally weighted regression (LWR) with radial basis functions and local polynomial fitting |
| [demo_manipulabilityTracking_mainTask01.m](./demos/demo_manipulabilityControl_mainTask01.m) | [[6]](#ref-6) | Tracking of a desired manipulability ellipsoid as the main task |
| [demo_manipulabilityTracking_mainTask02.m](./demos/demo_manipulabilityControl_mainTask02.m) | [[6]](#ref-6) | Same as demo_manipulabilityControl_mainTask01.m by using precision matrices as gain |
| [demo_manipulabilityTracking_secondaryTask01.m](./demos/demo_manipulabilityControl_secondTask01.m) | [[6]](#ref-6) | Tracking of a desired manipulability ellipsoid as the secondary task |
| [demo_manipulabilityTransfer01.m](./demos/demo_manipulabilityTransfer01.m) | [[6]](#ref-6) | Use of robot redundancy to track desired manipulability ellipsoid |
| [demo_manipulabilityTransfer02.m](./demos/demo_manipulabilityTransfer02.m) | [[6]](#ref-6) | Learning and reproduction of manipulability ellipsoid profiles |
| [demo_manipulabilityTransfer03.m](./demos/demo_manipulabilityTransfer03.m) | [[6]](#ref-6) | Learning and reproduction of manipulability ellipsoid profiles (numerical version) |
| [demo_OC_DDP_bicopter01.m](./demos/demo_OC_DDP_bicopter01.m) | [[11]](#ref-11) | iLQR applied to a bicopter problem |
| [demo_OC_DDP_car01.m](./demos/demo_OC_DDP_car01.m) | [[11]](#ref-11) | iLQR applied to a car parking problem |
| [demo_OC_DDP_cartpole01.m](./demos/demo_OC_DDP_cartpole01.m) | [[11]](#ref-11) | iLQR applied to a cartpole problem |
| [demo_OC_DDP_humanoid01.m](./demos/demo_OC_DDP_humanoid01.m) | [[11]](#ref-11) | iLQR applied to a planar 5-link humanoid problem with constraints between joints |
| [demo_OC_DDP_manipulator01.m](./demos/demo_OC_DDP_manipulator01.m) | [[11]](#ref-11) | iLQR applied to a 2D manipulator problem |
| [demo_OC_DDP_pendulum01.m](./demos/demo_OC_DDP_pendulum01.m) | [[11]](#ref-11) | iLQR applied to an actuated inverted pendulum problem |
| [demo_OC_LQT01.m](./demos/demo_OC_LQT01.m) | [[2]](#ref-2) | Batch solution of linear quadratic tracking (LQT) optimal control problem (example with viapoints and simple/double/triple integrator system) |
| [demo_OC_LQT02.m](./demos/demo_OC_LQT02.m) | [[2]](#ref-2) | Same as demo_OC_LQT01.m but with a GMM encoding of the reference (example by tracking position and velocity reference) |
| [demo_OC_LQT03.m](./demos/demo_OC_LQT03.m) | [[2]](#ref-2) | Same as demo_OC_LQT01.m but with a GMM encoding of the reference (example by tracking only position reference) |
| [demo_OC_LQT04.m](./demos/demo_OC_LQT04.m) | [[2]](#ref-2) | Control of a spring attached to a point with batch LQR (with augmented state space) |
| [demo_OC_LQT_augmSigma01.m](./demos/demo_OC_LQT_augmSigma01.m) | [[2]](#ref-2) | Batch LQR with augmented covariance to transform a tracking problem to a regulation problem |
| [demo_OC_LQT_ballistic01.m](./demos/demo_OC_LQT_ballistic01.m) | [[2]](#ref-2) | Batch LQT with augmented state to solve simple ballistic problem |
| [demo_OC_LQT_constrained01.m](./demos/demo_OC_LQT_constrained01.m) | [[2]](#ref-2) | Constrained batch LQT by using quadratic programming solver, with an encoding of position and velocity data |
| [demo_OC_LQT_fullQ01.m](./demos/demo_OC_LQT_fullQ01.m) | [[2]](#ref-2) | Batch LQT problem exploiting full Q matrix by constraining the motion to pass through a common point at different time steps |
| [demo_OC_LQT_fullQ02.m](./demos/demo_OC_LQT_fullQ02.m) | [[2]](#ref-2) | Batch LQT problem exploiting full Q matrix by constraining the motion of two agents |
| [demo_OC_LQT_fullQ03.m](./demos/demo_OC_LQT_fullQ03.m) | [[2]](#ref-2) | Batch LQT problem exploiting full Q matrix by constraining the motion of two agents in a simple ballistic task |
| [demo_OC_LQT_fullQ04.m](./demos/demo_OC_LQT_fullQ04.m) | [[2]](#ref-2) | Batch LQT exploiting full Q matrix to constrain the motion of two agents in a ballistic task mimicking a bimanual tennis serve problem |
| [demo_OC_LQT_infHor01.m](./demos/demo_OC_LQT_infHor01.m) | [[2]](#ref-2) | Discrete infinite horizon linear quadratic regulation (with precision matrix only on position) |
| [demo_OC_LQT_infHor02.m](./demos/demo_OC_LQT_infHor02.m) | [[2]](#ref-2) | Discrete infinite horizon linear quadratic regulation (with precision matrix on position and velocity) |
| [demo_OC_LQT_infHor03.m](./demos/demo_OC_LQT_infHor03.m) | [[2]](#ref-2) | Continuous infinite horizon linear quadratic tracking, by relying on a GMM encoding of position and velocity data |
| [demo_OC_LQT_infHor04.m](./demos/demo_OC_LQT_infHor04.m) | [[2]](#ref-2) | Discrete infinite horizon linear quadratic tracking, by relying on a GMM encoding of position and velocity data |
| [demo_OC_LQT_recursive01.m](./demos/demo_OC_LQT_recursive01.m) | [[2]](#ref-2) | Recursive computation of linear quadratic tracking (with feedback and feedforward terms) |
| [demo_OC_LQT_recursive02.m](./demos/demo_OC_LQT_recursive02.m) | [[2]](#ref-2) | Same as demo_OC_LQT_recursive01.m, by relying on a GMM encoding of position and velocity data, including comparison with batch LQT |
| [demo_OC_LQT_recursive03.m](./demos/demo_OC_LQT_recursive03.m) | [[2]](#ref-2) | Same as demo_OC_LQT_recursive01.m, by relying on a GMM encoding of only position data |
| [demo_OC_LQT_recursive_augmSigma01.m](./demos/demo_OC_LQT_recursive_augmSigma01.m) | [[2]](#ref-2) | Recursive LQR with augmented covariance to transform the tracking problem to a regulation problem |
| [demo_OC_LQT_Lagrangian01.m](./demos/demo_OC_LQT_Lagrangian01.m) | [[2]](#ref-2) | Batch LQR with Lagrangian in matrix form to force first and last point to coincide in order to form periodic motion |
| [demo_OC_LQT_noInitialState01.m](./demos/demo_OC_LQT_noInitialState01.m) | [[2]](#ref-2) | Batch LQR solution finding the optimal initial state together with the optimal control commands |
| [demo_OC_LQT_nullspace01.m](./demos/demo_OC_LQT_nullspace01.m) | [[2]](#ref-2) | Batch LQR with nullspace formulation |
| [demo_OC_LQT_nullspace02.m](./demos/demo_OC_LQT_nullspace02.m) | [[2]](#ref-2) | Batch LQR with nullspace formulation - ballistic task with an augmented state space |
| [demo_OC_LQT_online01.m](./demos/demo_OC_LQT_online01.m) | [[2]](#ref-2) | MPC recomputed in an online manner with a time horizon |
| [demo_OC_LQT_online02.m](./demos/demo_OC_LQT_online02.m) | [[2]](#ref-2) | MPC recomputed in an online manner with a time horizon, by relying on a GMM encoding of position and velocity data (with animation) |
| [demo_OC_LQT_online03.m](./demos/demo_OC_LQT_online03.m) | [[2]](#ref-2) | Obstacle avoidance with MPC recomputed in an online manner |
| [demo_OC_LQT_online_minimal01.m](./demos/demo_OC_LQT_online_minimal01.m) | [[2]](#ref-2) | Minimal example of online batch LQT |
| [demo_OC_LQT_skillsRepr01.m](./demos/demo_OC_LQT_skillsRepr01.m) | [[2]](#ref-2) | Representation of skills combined in parallel and in series through a batch LQT formulation |
| [demo_OC_LQT_viapoints01.m](./demos/demo_OC_LQT_viapoints01.m) | [[2]](#ref-2) | Keypoint-based motion through MPC, with a GMM encoding of position and velocity |
| [demo_OC_LQT_viapoints02.m](./demos/demo_OC_LQT_viapoints02.m) | [[2]](#ref-2) | Same as demo_OC_LQT_viapoints01 with only position encoding |
| [demo_OC_LQT_viapoints03.m](./demos/demo_OC_LQT_viapoints03.m) | [[2]](#ref-2) | Equivalence between cubic Bezier curve and batch LQT with double integrator |
| [demo_OC_LQT_viapoints_withProd01.m](./demos/demo_OC_LQT_viapoints_withProd01.m) | [[2]](#ref-2) | Batch LQT with viapoints computed as a product of trajectory distributions in control space |
| [demo_PCA01.m](./demos/demo_PCA01.m) | [[10]](#ref-10) | Principal component analysis (PCA) |
| [demo_Procrustes01.m](./demos/demo_Procrustes01.m) | [[10]](#ref-10) | SVD solution of orthogonal Procrustes problem |
| [demo_proMP01.m](./demos/demo_proMP01.m) | [[5]](#ref-5) | Conditioning on trajectory distributions with Probabilistic movement primitives to estimate trajectory distribution|
| [demo_proMP_Fourier01.m](./demos/demo_proMP_Fourier01.m) | [[5]](#ref-5) | ProMP with Fourier basis functions (1D example) |
| [demo_proMP_Fourier02.m](./demos/demo_proMP_Fourier02.m) | [[5]](#ref-5) | ProMP with Fourier basis functions (2D example) |
| [demo_proMP_Fourier_sampling01.m](./demos/demo_proMP_Fourier_sampling01.m) | [[5]](#ref-5) | Stochastic sampling with Fourier movement primitives (1D example) |
| [demo_proMP_Fourier_sampling02.m](./demos/demo_proMP_Fourier_sampling02.m) | [[5]](#ref-5) | Stochastic sampling with Fourier movement primitives (2D example) |
| [demo_regularization01.m](./demos/demo_regularization01.m) | [[10]](#ref-10) | Regularization of GMM parameters with minimum admissible eigenvalue |
| [demo_regularization02.m](./demos/demo_regularization02.m) | [[10]](#ref-10) | Regularization of GMM parameters with the addition of a small circular covariance |
| [demo_Riemannian_Gdp_vecTransp01.m](./demos/demo_Riemannian_Gdp_vecTransp01.m) | [[3]](#ref-3) | Parallel transport on Grassmann manifold |
| [demo_Riemannian_Hd_GMM01.m](./demos/demo_Riemannian_Hd_GMM01.m) | [[3]](#ref-3) | GMM on d-hyperboloid manifold |
| [demo_Riemannian_Hd_interp01.m](./demos/demo_Riemannian_Hd_interp01.m) | [[3]](#ref-3) | Interpolation on d-hyperboloid manifold |
| [demo_Riemannian_pose_GMM01.m](./demos/demo_Riemannian_pose_GMM01.m) | [[3]](#ref-3) | GMM to encode 3D position and orientation as unit quaternion by relying on Riemannian manifold |
| [demo_Riemannian_S1_interp01.m](./demos/demo_Riemannian_S1_interp01.m) | [[3]](#ref-3) | Interpolation on 1-sphere manifold (formulation with imaginary numbers) |
| [demo_Riemannian_S1_interp02.m](./demos/demo_Riemannian_S1_interp02.m) | [[3]](#ref-3) | Interpolation on 1-sphere manifold (formulation with imaginary numbers, parameterization of x as angle) |
| [demo_Riemannian_S2_batchLQR01.m](./demos/demo_Riemannian_S2_batchLQR01.m) | [[3]](#ref-3) | LQT on a sphere by relying on Riemannian manifold, based on GMM encoding of movement |
| [demo_Riemannian_S2_batchLQR02.m](./demos/demo_Riemannian_S2_batchLQR02.m) | [[3]](#ref-3) | Same as demo_Riemannian_S2_batchLQR01.m with full reference |
| [demo_Riemannian_S2_batchLQR03.m](./demos/demo_Riemannian_S2_batchLQR03.m) | [[3]](#ref-3) | Same as demo_Riemannian_S2_batchLQR02.m by using only position data (-> velocity commands) |
| [demo_Riemannian_S2_batchLQR_Bezier01.m](./demos/demo_Riemannian_S2_batchLQR_Bezier01.m) | [[8]](#ref-8) | Bezier interpolation on a sphere by relying on Riemannian manifold and batch LQR |
| [demo_Riemannian_S2_GaussProd01.m](./demos/demo_Riemannian_S2_GaussProd01.m) | [[3]](#ref-3) | Product of Gaussians on a sphere by relying on Riemannian manifold |
| [demo_Riemannian_S2_GMM01.m](./demos/demo_Riemannian_S2_GMM01.m) | [[3]](#ref-3) | GMM for data on a sphere by relying on Riemannian manifold |
| [demo_Riemannian_S2_GMR01.m](./demos/demo_Riemannian_S2_GMR01.m) | [[3]](#ref-3) | GMR with input and output data on a sphere by relying on Riemannian manifold |
| [demo_Riemannian_S2_GMR02.m](./demos/demo_Riemannian_S2_GMR02.m) | [[3]](#ref-3) | GMR with time as input and spherical data as output by relying on Riemannian manifold |
| [demo_Riemannian_S2_GMR03.m](./demos/demo_Riemannian_S2_GMR03.m) | [[3]](#ref-3) | GMR with 3D Euclidean data as input and spherical data as output by relying on Riemannian manifold |
| [demo_Riemannian_S2_GMR04.m](./demos/demo_Riemannian_S2_GMR04.m) | [[3]](#ref-3) | GMR with input data on a sphere and output data in Eudlidean space by relying on Riemannian manifold |
| [demo_Riemannian_S2_infHorLQR01.m](./demos/demo_Riemannian_S2_infHorLQR01.m) | [[3]](#ref-3) | Linear quadratic regulation on a sphere by relying on Riemannian manifold and infinite-horizon LQR |
| [demo_Riemannian_S2_TPGMM01.m](./demos/demo_Riemannian_S2_TPGMM01.m) | [[3]](#ref-3) | TP-GMM for data on a sphere by relying on Riemannian manifold (with single frame) |
| [demo_Riemannian_S2_TPGMM02.m](./demos/demo_Riemannian_S2_TPGMM02.m) | [[3]](#ref-3) | TP-GMM for data on a sphere by relying on Riemannian manifold (with two frames) |
| [demo_Riemannian_S2_vecTransp01.m](./demos/demo_Riemannian_S2_vecTransp01.m) | [[3]](#ref-3) | Parallel transport on a sphere |
| [demo_Riemannian_S3_GMM01.m](./demos/demo_Riemannian_S3_GMM01.m) | [[3]](#ref-3) | GMM for unit quaternion data by relying on Riemannian manifold |
| [demo_Riemannian_S3_GMR01.m](./demos/demo_Riemannian_S3_GMR01.m) | [[3]](#ref-3) | GMR with unit quaternions as input and output data by relying on Riemannian manifold |
| [demo_Riemannian_S3_GMR02.m](./demos/demo_Riemannian_S3_GMR02.m) | [[3]](#ref-3) | GMR with time as input and unit quaternion as output by relying on Riemannian manifold |
| [demo_Riemannian_S3_infHorLQR01.m](./demos/demo_Riemannian_S3_infHorLQR01.m) | [[3]](#ref-3) | Linear quadratic regulation of unit quaternions by relying on Riemannian manifold and infinite-horizon LQR |
| [demo_Riemannian_S3_interp01.m](./demos/demo_Riemannian_S3_interp01.m) | [[3]](#ref-3) | Interpolation of unit quaternions by relying on Riemannian manifold, providing the same result as SLERP interpolation |
| [demo_Riemannian_S3_vecTransp01.m](./demos/demo_Riemannian_S3_vecTransp01.m) | [[3]](#ref-3) | Parallel transport for unit quaternions |
| [demo_Riemannian_Sd_GaussProd01.m](./demos/demo_Riemannian_Sd_GaussProd01.m) | [[3]](#ref-3) | Product of Gaussians on a d-sphere by relying on Riemannian manifold |
| [demo_Riemannian_Sd_GMM01.m](./demos/demo_Riemannian_Sd_GMM01.m) | [[3]](#ref-3) | GMM for data on a 1-sphere (circle) by relying on Riemannian manifold |
| [demo_Riemannian_Sd_GMM02.m](./demos/demo_Riemannian_Sd_GMM02.m) | [[3]](#ref-3) | GMM for data on a 2-sphere (ball) by relying on Riemannian manifold |
| [demo_Riemannian_Sd_GMMR01.m](./demos/demo_Riemannian_Sd_GMR01.m) | [[3]](#ref-3) | Retrieval of periodic motion with GMR, by considering input data on a 1-sphere and output data in Euclidean space |
| [demo_Riemannian_Sd_GMMR02.m](./demos/demo_Riemannian_Sd_GMR02.m) | [[3]](#ref-3) | GMR with input data on a 2-sphere (ball) and output data in Euclidean space by relying on Riemannian manifold |
| [demo_Riemannian_Sd_interp01.m](./demos/demo_Riemannian_Sd_interp01.m) | [[3]](#ref-3) | Interpolation on a n-sphere (formulation with tangent space of same dimension as manifold) |
| [demo_Riemannian_Sd_interp02.m](./demos/demo_Riemannian_Sd_interp02.m) | [[3]](#ref-3) | Interpolation on a 3-sphere and comparison with SLERP |
| [demo_Riemannian_Sd_MPC01.m](./demos/demo_Riemannian_Sd_MPC01.m) | [[3]](#ref-3) | LQT on Sn by relying on Riemannian manifold and batch LQR recomputed in an online manner, based on GMM encoding of movement |
| [demo_Riemannian_Sd_MPC_infHor01.m](./demos/demo_Riemannian_Sd_MPC_infHor01.m) | [[3]](#ref-3) | Linear quadratic regulation on S3 by relying on Riemannian manifold and infinite-horizon LQR |
| [demo_Riemannian_Sd_vecTransp01.m](./demos/demo_Riemannian_Sd_vecTransp01.m) | [[3]](#ref-3) | Parallel transport on a n-sphere |
| [demo_Riemannian_Sd_vecTransp02.m](./demos/demo_Riemannian_Sd_vecTransp02.m) | [[3]](#ref-3) | Vector transport on a n-sphere using Schild's ladder algorithm |
| [demo_Riemannian_SE2_GMM01.m](./demos/demo_Riemannian_SE2_GMM01.m) | [[3]](#ref-3) | GMM on SE(2) manifold |
| [demo_Riemannian_SE2_interp01.m](./demos/demo_Riemannian_SE2_interp01.m) | [[3]](#ref-3) | Interpolation on SE(2) manifold |
| [demo_Riemannian_SOd_interp01.m](./demos/demo_Riemannian_SOd_interp01.m) | [[3]](#ref-3) | Interpolation on SO(d) manifold |
| [demo_Riemannian_SPD_GMM01.m](./demos/demo_Riemannian_SPD_GMM01.m) | [[4]](#ref-4) | GMM for covariance data by relying on Riemannian manifold |
| [demo_Riemannian_SPD_GMM_augmSigma01.m](./demos/demo_Riemannian_SPD_GMM_augmSigma01.m) | [[4]](#ref-4) | GMM to encode ellipsoid datapoints (centers and covariance matrices) by relying on augmented covariance embeddings and Riemannian manifold |
| [demo_Riemannian_SPD_GMM_tensor01.m](./demos/demo_Riemannian_SPD_GMM_tensor01.m) | [[4]](#ref-4) | GMM for covariance data by relying on Riemannian manifold |
| [demo_Riemannian_SPD_GMR01.m](./demos/demo_Riemannian_SPD_GMR01.m) | [[4]](#ref-4) | GMR with time as input and covariance data as output by relying on Riemannian manifold |
| [demo_Riemannian_SPD_GMR02.m](./demos/demo_Riemannian_SPD_GMR02.m) | [[4]](#ref-4) | GMR with time as input and position vector as output with comparison between computation in vector and matrix forms |
| [demo_Riemannian_SPD_GMR03.m](./demos/demo_Riemannian_SPD_GMR03.m) | [[4]](#ref-4) | GMR with vector as input and covariance data as output by relying on Riemannian manifold |
| [demo_Riemannian_SPD_GMR_tensor01.m](./demos/demo_Riemannian_SPD_GMR_tensor01.m) | [[4]](#ref-4) | GMR with time as input and covariance data as output by relying on Riemannian manifold |
| [demo_Riemannian_SPD_interp01.m](./demos/demo_Riemannian_SPD_interp01.m) | [[4]](#ref-4) | Covariance interpolation on Riemannian manifold (comparison with linear interpolation, Euclidean interpolation on Cholesky decomposition, and Wasserstein interpolation) |
| [demo_Riemannian_SPD_interp02.m](./demos/demo_Riemannian_SPD_interp02.m) | [[4]](#ref-4) | Covariance interpolation on Riemannian manifold from a GMM with augmented covariances |
| [demo_Riemannian_SPD_interp03.m](./demos/demo_Riemannian_SPD_interp03.m) | [[4]](#ref-4) | Trajectory morphing through covariance interpolation on Riemannian manifold (with augmented Gaussian trajectory distribution) |
| [demo_Riemannian_SPD_vecTransp01.m](./demos/demo_Riemannian_SPD_vecTransp01.m) | [[4]](#ref-4) | Verification of angle conservation in parallel transport on the symmetric positive definite |
| [demo_Riemannian_SPD_vecTransp02.m](./demos/demo_Riemannian_SPD_vecTransp02.m) | [[4]](#ref-4) | Vector transport on the Symmetric Positive Definite matrices (SPD) manifold using Schild's ladder algorithm |
| [demo_search01.m](./demos/demo_search01.m) | [[2]](#ref-2) | EM-based stochastic optimization |
| [demo_spring01.m](./demos/demo_spring01.m) | [[10]](#ref-10) | Influence of the damping ratio in mass-spring-damper systems |
| [demo_stdPGMM01.m](./demos/demo_stdPGMM01.m) | [[1]](#ref-1) | Parametric Gaussian mixture model (PGMM) used as a task-parameterized model, with DS-GMR employed to retrieve continuous movements |
| [demo_tensor_TTGO01.m](./demos/demo_tensor_TTGO01.m) | [[12]](#ref-12) | Global optimization with tensor trains (TTGO) |
| [demo_TPHDDC01.m](./demos/demo_TPHDDC01.m) | [[1]](#ref-1) | Task-parameterized high dimensional data clustering (TP-HDDC) |
| [demo_TPGMM01.m](./demos/demo_TPGMM01.m) | [[1]](#ref-1) | Task-parameterized Gaussian mixture model (TP-GMM) encoding |
| [demo_TPGMM_bimanualReaching01.m](./demos/demo_TPGMM_bimanualReaching01.m) | [[1]](#ref-1) | Time-invariant task-parameterized GMM applied to a bimanual reaching task |
| [demo_TPGMM_teleoperation01.m](./demos/demo_TPGMM_teleoperation01.m) | [[1]](#ref-1) | Time-invariant task-parameterized GMM applied to a teleoperation task (position only) |
| [demo_TPGMR01.m](./demos/demo_TPGMR01.m) | [[1]](#ref-1) | Task-parameterized Gaussian mixture model (TP-GMM), with GMR used for reproduction (without dynamical system) |
| [demo_TPGP01.m](./demos/demo_TPGP01.m) | [[1]](#ref-1) | Task-parameterized Gaussian process regression (TP-GPR) |
| [demo_TPLQT01.m](./demos/demo_TPLQT01.m) | [[1]](#ref-1) | Task-parameterized probabilistic model encoding position data, with LQT used to track the associated stepwise reference path |
| [demo_TPMFA01.m](./demos/demo_TPMFA01.m) | [[1]](#ref-1) | Task-parameterized mixture of factor analyzers (TP-MFA) |
| [demo_TPMPPCA01.m](./demos/demo_TPMPPCA01.m) | [[1]](#ref-1) | Task-parameterized mixture of probabilistic principal component analyzers (TP-MPPCA) |
| [demo_TPproMP01.m](./demos/demo_TPproMP01.m) | [[1]](#ref-1) | Task-parameterized probabilistic movement primitives (TP-ProMP) |
| [demo_TPtrajDistrib01.m](./demos/demo_TPtrajDistrib01.m) | [[1]](#ref-1) | Task-parameterized model with trajectory distribution and eigendecomposition |
| [demo_TPtrajGMM01.m](./demos/demo_TPtrajGMM01.m) | [[1]](#ref-1) | Task-parameterized model with trajectory-GMM encoding |
| [demo_trajDistrib01.m](./demos/demo_trajDistrib01.m) | [[2]](#ref-2) | Stochastic sampling with Gaussian trajectory distribution |
| [demo_trajDistrib_differencingMatrix01.m](./demos/demo_trajDistrib_differencingMatrix01.m) | [[2]](#ref-2) | Conditioning on trajectory distribution constructed by differencing matrix, with via-point passing example |
| [demo_trajGMM01.m](./demos/demo_trajGMM01.m) | [[2]](#ref-2) | Trajectory synthesis using a GMM with dynamic features (trajectory GMM) |
| [demo_trajGMM02.m](./demos/demo_trajGMM02.m) | [[2]](#ref-2) | Trajectory synthesis with a GMM with dynamic features (trajectory GMM), where the GMM is learned from trajectory examples ||
| [demo_trajHSMM01.m](./demos/demo_trajHSMM01.m) | [[2]](#ref-2) | Trajectory synthesis with an HSMM with dynamic features (trajectory-HSMM) |
| [demo_trajHSMM_adaptiveDuration01.m](./demos/demo_trajHSMM_adaptiveDuration01.m) | [[9]](#ref-9) | Hidden semi-Markov model with adaptive duration |
| [demo_trajHSMM_adaptiveDuration_online01.m](./demos/demo_trajHSMM_adaptiveDuration_online01.m) | [[9]](#ref-9) | Online trajectory retrieval method built on an HSMM with adaptive duration and a trajectory-GMM representation |
| [benchmarks/benchmark_DS_GP_GMM01.m](./demos/benchmarks/benchmark_DS_GP_GMM01.m) | [[1]](#ref-1) | Benchmark of task-parameterized model based on GPR, with trajectory model (GMM encoding) |
| [benchmarks/benchmark_DS_GP_raw01.m](./demos/benchmarks/benchmark_DS_GP_raw01.m) | [[1]](#ref-1) | Benchmark of task-parameterized model based on GPR, with raw trajectory |
| [benchmarks/benchmark_DS_PGMM01.m](./demos/benchmarks/benchmark_DS_PGMM01.m) | [[1]](#ref-1) | Benchmark of task-parameterized model based on parametric Gaussian mixture model |
| [benchmarks/benchmark_DS_TP_GMM01.m](./demos/benchmarks/benchmark_DS_TP_GMM01.m) | [[1]](#ref-1) | Benchmark of task-parameterized Gaussian mixture model (TP-GMM) |
| [benchmarks/benchmark_DS_TP_GP01.m](./demos/benchmarks/benchmark_DS_TP_GP01.m) | [[1]](#ref-1) | Benchmark of task-parameterized Gaussian process (nonparametric task-parameterized method) |
| [benchmarks/benchmark_DS_TP_LWR01.m](./demos/benchmarks/benchmark_DS_TP_LWR01.m) | [[1]](#ref-1) | Benchmark of task-parameterized locally weighted regression (nonparametric task-parameterized method) |
| [benchmarks/benchmark_DS_TP_MFA01.m](./demos/benchmarks/benchmark_DS_TP_MFA01.m) | [[1]](#ref-1) | Benchmark of task-parameterized mixture of factor analyzers (TP-MFA) |
| [benchmarks/benchmark_DS_TP_trajGMM01.m](./demos/benchmarks/benchmark_DS_TP_trajGMM01.m) | [[1]](#ref-1) | Benchmark of task-parameterized Gaussian mixture model (TP-GMM) |


### References

If you find PbDlib useful for your research, please cite the related publications!

<p><a name="ref-1">
[1] Calinon, S. (2016). <strong>A Tutorial on Task-Parameterized Movement Learning and Retrieval</strong>. Intelligent Service Robotics (Springer), 9:1, 1-29.
</a><br>
[[pdf]](https://calinon.ch/papers/Calinon-JIST2015.pdf)
[[bib]](https://calinon.ch/papers/Calinon-JIST2015.bib)
<br><strong>(Ref. for GMM, TP-GMM, MFA, MPPCA, GPR, trajGMM)</strong>
</p>

<p><a name="ref-2">
[2] Calinon, S. and Lee, D. (2019). <strong>Learning Control</strong>. Vadakkepat, P. and Goswami, A. (eds.). Humanoid Robotics: a Reference, pp. 1261-1312. Springer.
</a><br>
[[pdf]](https://calinon.ch/papers/Calinon-Lee-learningControl.pdf)
[[bib]](https://calinon.ch/papers/Calinon-Lee-learningControl.bib)
<br><strong>(Ref. for LQR, MPC, HMM, HSMM)</strong>
</p>

<p><a name="ref-3">
[3] Calinon, S. (2020). <strong>Gaussians on Riemannian Manifolds: Applications for Robot Learning and Adaptive Control</strong>. IEEE Robotics and Automation Magazine (RAM), 27:2, 33-45.
</a><br>
[[pdf]](https://calinon.ch/papers/Calinon-RAM2020.pdf)
[[bib]](https://calinon.ch/papers/Calinon-RAM2020.bib)
<br><strong>(Ref. for Riemannian manifolds)</strong>
</p>

<p><a name="ref-4">
[4] Jaquier, N. and Calinon, S. (2017). <strong>Gaussian Mixture Regression on Symmetric Positive Definite Matrices Manifolds: Application to Wrist Motion Estimation with sEMG</strong>. In Proc. of the IEEE/RSJ Intl Conf. on Intelligent Robots and Systems (IROS), pp. 59-64.
</a><br>
[[pdf]](https://calinon.ch/papers/Jaquier-IROS2017.pdf)
[[bib]](https://calinon.ch/papers/Jaquier-IROS2017.bib)
<br><strong>(Ref. for S^+ Riemannian manifolds)</strong>
</p>

<p><a name="ref-5">
[5] Calinon, S. (2019). <strong>Mixture Models for the Analysis, Edition, and Synthesis of Continuous Time Series</strong>. Bouguila, N. and Fan, W. (eds). Mixture Models and Applications, pp. 39-57. Springer.
</a><br>
[[pdf]](https://calinon.ch/papers/Calinon_MMchapter2019.pdf)
[[bib]](https://calinon.ch/papers/Calinon_MMchapter2019.bib)
<br><strong>(Ref. for ergodic control, Bezier curves, LWR, GMR, probabilistic movement primitives)</strong>
</p>

<p><a name="ref-6"> 
[6] Jaquier, N., Rozo, L., Caldwell, D.G. and Calinon, S. (2020). <strong>Geometry-aware Manipulability Learning, Tracking and Transfer</strong>. International Journal of Robotics Research (IJRR).
</a><br>
[[pdf]](https://calinon.ch/papers/Jaquier-IJRR2020.pdf)
[[bib]](https://calinon.ch/papers/Jaquier-IJRR2020.bib)
<br><strong>(Ref. for manipulability ellipsoids)</strong>
</p>

<p><a name="ref-7">
[7] Bruno, D., Calinon, S. and Caldwell, D.G. (2017). <strong>Learning Autonomous Behaviours for the Body of a Flexible Surgical Robot</strong>. Autonomous Robots, 41:2, 333-347.
</a><br>
[[pdf]](https://calinon.ch/papers/Bruno-AURO2017.pdf)
[[bib]](https://calinon.ch/papers/Bruno-AURO2017.bib)
<br><strong>(Ref. for DP-means)</strong>
</p>

<p><a name="ref-8">
[8] Berio, D., Calinon, S. and Fol Leymarie, F. (2017). <strong>Generating Calligraphic Trajectories with Model Predictive Control</strong>. In Proc. of the 43rd Conf. on Graphics Interface, pp. 132-139.
</a><br>
[[pdf]](https://calinon.ch/papers/Berio-GI2017.pdf)
[[bib]](https://calinon.ch/papers/Berio-GI2017.bib)
<br><strong>(Ref. for Bezier curves as LQT with viapoints)</strong>
</p>

<p><a name="ref-9">
[9] Rozo, L., Silvério, J., Calinon, S. and Caldwell, D.G. (2016). <strong>Learning Controllers for Reactive and Proactive Behaviors in Human-Robot Collaboration</strong>. Frontiers in Robotics and AI, 3:30, 1-11.
</a><br>
[[pdf]](https://calinon.ch/papers/Rozo-Frontiers2016.pdf)
[[bib]](https://calinon.ch/papers/Rozo-Frontiers2016.bib)
<br><strong>(Ref. for HSMM with adaptive time duration)</strong>
</p>

<p><a name="ref-10">
[10] EPFL EE613 course "Machine Learning for Engineers"
</a><br>
[[url]](https://calinon.ch/teaching_EPFL.htm)
<br><strong>(Ref. for machine learning teaching material)</strong>
</p>

<p><a name="ref-11">
[11] Lembono, T.S. and Calinon, S. (2021). <strong>Probabilistic Iterative LQR for Short Time Horizon MPC</strong>. In Proc. IEEE/RSJ Intl Conf. on Intelligent Robots and Systems (IROS), pp. 556-562.
</a><br>
[[pdf]](https://calinon.ch/papers/Lembono-IROS2021.pdf)
[[bib]](https://calinon.ch/papers/Lembono-IROS2021.bib)
<br><strong>(Ref. for iLQR/DDP)</strong>
</p>

<p><a name="ref-12">
[12] Shetty, S., Lembono, T., Löw, T. and Calinon, S. (2022). <strong>Tensor Train for Global Optimization Problems in Robotics</strong>. arXiv:2206.05077.
</a><br>
[[pdf]](https://calinon.ch/papers/Shetty-arXiv2022.pdf)
[[bib]](https://calinon.ch/papers/Shetty-arXiv2022.bib)
<br><strong>(Ref. for low-rank tensor train factorization)</strong>
</p>

### Gallery

|                         |                         |
|-------------------------|-------------------------|
| ![](https://gitlab.idiap.ch/rli/pbdlib-matlab/raw/master/images/demo_GMM_semiTied01.png) <br> [demo\_GMM\_semiTied01.m](./demos/demo_GMM_semiTied01.m) | ![](https://gitlab.idiap.ch/rli/pbdlib-matlab/raw/master/images/demo_GMR_3Dviz01.png) <br> [demo\_GMR\_3Dviz01.m](./demos/demo_GMR_3Dviz01.m) |
| ![](https://gitlab.idiap.ch/rli/pbdlib-matlab/raw/master/images/demo_GMR_polyFit01.png) <br> [demo\_GMR\_polyFit01.m](./demos/demo_GMR_polyFit01.m) | ![](https://gitlab.idiap.ch/rli/pbdlib-matlab/raw/master/images/demo_HMM01.png) <br> [demo\_HMM01.m](./demos/demo_HMM01.m) |
| ![](https://gitlab.idiap.ch/rli/pbdlib-matlab/raw/master/images/demo_MPC_iterativeLQR01.png) <br> [demo\_OC\_LQT\_recursive02.m](./demos/demo_OC_LQT_recursive02.m) | ![](https://gitlab.idiap.ch/rli/pbdlib-matlab/raw/master/images/demo_Riemannian_SPD_GMR01.png) <br> [demo\_Riemannian\_SPD\_GMR01.m](./demos/demo_Riemannian_SPD_GMR01.m) |
| ![](https://gitlab.idiap.ch/rli/pbdlib-matlab/raw/master/images/demo_Riemannian_SPD_interp01.png) <br> [demo\_Riemannian\_SPD\_interp01.m](./demos/demo_Riemannian_SPD_interp01.m) | ![](https://gitlab.idiap.ch/rli/pbdlib-matlab/raw/master/images/demo_Riemannian_SPD_interp02.png) <br> [demo\_Riemannian\_SPD\_interp02.m](./demos/demo_Riemannian_SPD_interp02.m) |
| ![](https://gitlab.idiap.ch/rli/pbdlib-matlab/raw/master/images/demo_Riemannian_SPD_interp03.png) <br> [demo\_Riemannian\_SPD\_interp03.m](./demos/demo_Riemannian_SPD_interp03.m) | ![](https://gitlab.idiap.ch/rli/pbdlib-matlab/raw/master/images/demo_Riemannian_S2_GMM01.png) <br> [demo\_Riemannian\_S2\_GMM01.m](./demos/demo_Riemannian_S2_GMM01.m) |
| ![](https://gitlab.idiap.ch/rli/pbdlib-matlab/raw/master/images/demo_TPMPC01.png) <br> [demo\_TPMPC01.m](./demos/demo_TPMPC01.m) | ![](https://gitlab.idiap.ch/rli/pbdlib-matlab/raw/master/images/demo_TPGMR01.png) <br> [demo\_TPGMR01.m](./demos/demo_TPGMR01.m) |
| ![](https://gitlab.idiap.ch/rli/pbdlib-matlab/raw/master/images/demo_TPproMP01.png) <br> [demo\_TPproMP01.m](./demos/demo_TPproMP01.m) | ![](https://gitlab.idiap.ch/rli/pbdlib-matlab/raw/master/images/demo_trajHSMM01.png) <br> [demo\_trajHSMM01.m](./demos/demo_trajHSMM01.m) |


### License

The Matlab/GNU Octave version of PbDlib is maintained by [Sylvain Calinon](https://calinon.ch), Idiap Research Institute. 

Copyright (c) 2015-2022 Idiap Research Institute, https://idiap.ch/

PbDlib is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 3 as published by the Free Software Foundation.

PbDlib is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with PbDlib. If not, see <https://www.gnu.org/licenses/>.
