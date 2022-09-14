function demo_manipulabilityTransfer02
% This code shows how a robot learns to follow a desired Cartesian trajectory while modifying its joint  
% configuration to match a desired profile of manipulability ellipsoids over time.
% The learning framework is built on two GMMs, one for encoding the demonstrated Cartesian trajectories, 
% and the other one for encoding the profiles of manipulability ellipsoids observed during the demonstrations.
% The former is a classic GMM, while the latter is a GMM that relies on an SPD-matrices manifold formulation.
%
% The demonstrations are generated with a 3-DoFs planar robot that follows a set of Cartesian trajectories. 
% The reproduction is carried out by a 5-DoFs planar robot.
%
% For this example, time is input for both models, while for GMM1 and GMM2 the outputs respectively correspond
% to a 2D Cartesian position and a manipulability ellipsoid (3x3 positive definite matrix).
%
% This demo requires the robotics toolbox RTB10 (http://petercorke.com/wordpress/toolboxes/robotics-toolbox).
% First run 'startup_rvc' from the robotics toolbox.
%
% If this code is useful for your research, please cite the related publication:
% @article{Jaquier18,
% 	author="Jaquier, N. and Rozo, L. and Caldwell, D. G. and Calinon, S.",
% 	title="Geometry-aware Manipulability Transfer",
% 	journal="arXiv:1811.11050",
% 	year="2018",
% 	pages="1--20"
% }
%
% Written by Leonel Rozo, 2017
%
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
%
% PbDlib is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License version 3 as
% published by the Free Software Foundation.
%
% PbDlib is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with PbDlib. If not, see <http://www.gnu.org/licenses/>.

addpath('./m_fcts/');
disp('This demo requires the robotics toolbox RTB10 (http://petercorke.com/wordpress/toolboxes/robotics-toolbox).');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbData = 100; % Number of datapoints in a trajectory
nbSamples = 4; % Number of demonstrations
nbIter = 10; % Number of iteration for the Gauss Newton algorithm (Riemannian manifold)
nbIterEM = 10; % Number of iteration for the EM algorithm
nbIterME = 30; % Number of iterations for the redundancy resolution
letter = 'L'; % Letter to use as dataset for demonstration data
% letter = 'I'; % Letter to use as dataset for demonstration data
typeCost = 1; % Type of cost function for manipulability transfer
alpha = 50; % Gain for cost gradient
% typeCost = 2; % Type of cost function for manipulability transfer
% alpha = 12e0; % Gain for cost gradient in nullspace velocities

modelPD.nbStates = 4; %Number of Gaussians in the GMM over man. ellipsoids
modelPD.nbVar = 3; % Dimension of the manifold and tangent space (1D input + 2^2 output)
modelPD.nbVarCovOut = modelPD.nbVar + modelPD.nbVar*(modelPD.nbVar-1)/2; %Dimension of the output covariance
modelPD.dt = 1E-2; % Time step duration
modelPD.params_diagRegFact = 1E-4; % Regularization of covariance
modelPD.Kp = 120; % Gain for position control in task space

modelKin.nbStates = 4; % Number of states in the GMM over 2D Cartesian trajectories
modelKin.nbVar = 3; % Number of variables [t,x1,x2]
modelKin.dt = modelPD.dt; % Time step duration

% Code control variables
needsLearning = 1;
saveModel = 0;

% Initialisation of the covariance transformation for GMM over man. ellips.
[covOrder4to2, covOrder2to4] = set_covOrder4to2(modelPD.nbVar);

% Tensor regularization term
tensor_diagRegFact_mat = eye(modelPD.nbVar + modelPD.nbVar * (modelPD.nbVar-1)/2);
tensor_diagRegFact = covOrder2to4(tensor_diagRegFact_mat);


%% Create robots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robots parameters
nbDOFs = 3; % Nb of degrees of freedom for teacher robot
nbDOFt = nbDOFs+2; % Nb of degrees of freedom for student robot
armLength = 4;
L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
robotT = SerialLink(repmat(L1,nbDOFs,1)); % Robot teacher
robotS = SerialLink(repmat(L1,nbDOFt,1)); % Robot student
q0T = [-pi/2 0.0 pi/3]; % Initial robot configuration

%% Symbolic Jacobian and manipulability ellipsoid
% Symbolic Jacobian and VME:
qSym = sym('q', [1 robotS.n]);	% Symbolic robot joints
J_Rs = robotS.jacob0(qSym.');
ME_c = J_Rs(1:2,:)*J_Rs(1:2,:)';	% Current VME for planar case (x,y)


%% Load handwriting data and generating manipulability ellipsoids
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(needsLearning)
	disp('Loading demonstration data...');
	load(['data/2Dletters/' letter '.mat'])
	
	xIn(1,:) = (1:nbData) * modelPD.dt; % Time as input variable
	X = zeros(3,3,nbData*nbSamples); % Matrix storing t,x1,x2 for all the demos
	X(1,1,:) = reshape(repmat(xIn,1,nbSamples),1,1,nbData*nbSamples); % Stores input
	Data=[];
	
	for n=1:nbSamples
		s(n).Data=[];
		
		dTmp = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
		if letter == 'I'
			dTmp = [dTmp(1,:) + 5; dTmp(2,:)]; % Translating original data
		end
		s(n).Data = [s(n).Data; dTmp];
		
		% Obtain robot configurations for the current demo given initial robot pose q0
		T = transl([s(n).Data(1:2,:) ; zeros(1,nbData)]');
		
		% One way to check robotics toolbox version
		if isobject(robotT.fkine(q0T)) % 10.X
			maskPlanarRbt = [ 1 1 0 0 0 0 ]; % Mask matrix for a 3-DoFs robots for position (x,y)
			q = robotT.ikine(T, q0T', 'mask', maskPlanarRbt)'; % Based on an initial pose
		else % 9.X
			maskPlanarRbt = [ 1 1 1 0 0 0 ];
			q = robotT.ikine(T, q0T', maskPlanarRbt)'; % Based on an initial pose
		end
		
		% Computing force/velocity manipulability ellipsoids, that will be later
		% used for encoding a GMM in the force/velocity manip. ellip. manifold
		for t = 1 : nbData
			auxJ = robotT.jacob0(q(:,t),'trans');
			J = auxJ(1:2,:);
			X(2:3,2:3,t+(n-1)*nbData) = J*J'; % Saving ME
		end
		Data = [Data [xIn ; s(n).Data]]; % Storing time and Cartesian positions
	end
	
	
	%% GMM parameters estimation
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	disp('Learning GMM1 (2D Cartesian position)...');
	modelKin = init_GMM_timeBased(Data, modelKin); % Model for position
	modelKin = EM_GMM(Data, modelKin);
	
	disp('Learning GMM2 (Manipulability ellipsoids)...');
	% Initialisation on the manifold
	modelPD = spd_init_GMM_kbins(X, modelPD, nbSamples); % Model for Man. Ellips.
	modelPD.Mu = zeros(size(modelPD.MuMan));
	in=1; % Input dimension
	out=2:modelPD.nbVar; % Output dimension
	id = cov_findNonZeroID(in, out, 0, 0);
	
	L = zeros(modelPD.nbStates, nbData*nbSamples);
	Xts = zeros(modelPD.nbVar, modelPD.nbVar, nbData*nbSamples, modelPD.nbStates);
	% EM for PD matrices manifold
	for nb=1:nbIterEM
		fprintf('.');
		
		% E-step
		for i=1:modelPD.nbStates
			Xts(in,in,:,i) = X(in,in,:)-repmat(modelPD.MuMan(in,in,i),1,1,nbData*nbSamples);
			Xts(out,out,:,i) = logmap(X(out,out,:), modelPD.MuMan(out,out,i));
			
			% Compute probabilities using the reduced form (computationally
			% less expensive than complete form)
			xts = symMat2Vec(Xts(:,:,:,i));
			MuVec = symMat2Vec(modelPD.Mu(:,:,i));
			SigmaVec = covOrder4to2(modelPD.Sigma(:,:,:,:,i));
			
			% 		L(i,:) = model.Priors(i) * gaussPDF2(xts, MuVec, SigmaVec);
			L(i,:) = modelPD.Priors(i) * gaussPDF(xts(id,:), MuVec(id,:), SigmaVec(id,id));
			
		end
		GAMMA = L ./ repmat(sum(L,1)+realmin, modelPD.nbStates, 1);
		H = GAMMA ./ repmat(sum(GAMMA,2)+realmin, 1, nbData*nbSamples);
		% M-step
		for i=1:modelPD.nbStates
			% Update Priors
			modelPD.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
			
			% Update MuMan
			for n=1:nbIter
				uTmpTot = zeros(modelPD.nbVar,modelPD.nbVar);
				uTmp = zeros(modelPD.nbVar,modelPD.nbVar,nbData*nbSamples);
				uTmp(in,in,:) = X(in,in,:)-repmat(modelPD.MuMan(in,in,i),1,1,nbData*nbSamples);
				uTmp(out,out,:) = logmap(X(out,out,:), modelPD.MuMan(out,out,i));
				
				for k = 1:nbData*nbSamples
					uTmpTot = uTmpTot + uTmp(:,:,k) .* H(i,k);
				end
				modelPD.MuMan(in,in,i) = uTmpTot(in,in) + modelPD.MuMan(in,in,i);
				modelPD.MuMan(out,out,i) = expmap(uTmpTot(out,out), modelPD.MuMan(out,out,i));
			end
			
			% Update SigmaMan
			modelPD.Sigma(:,:,:,:,i) = zeros(modelPD.nbVar,modelPD.nbVar,modelPD.nbVar,modelPD.nbVar);
			for k = 1:nbData*nbSamples
				modelPD.Sigma(:,:,:,:,i) = modelPD.Sigma(:,:,:,:,i) + H(i,k) .* outerprod(uTmp(:,:,k),uTmp(:,:,k));
			end
			modelPD.Sigma(:,:,:,:,i) = modelPD.Sigma(:,:,:,:,i) + tensor_diagRegFact.*modelPD.params_diagRegFact;
		end
	end
	
	% Eigendecomposition of Sigma
	for i=1:modelPD.nbStates
		[~, modelPD.V(:,:,:,i), modelPD.D(:,:,i)] = covOrder4to2(modelPD.Sigma(:,:,:,:,i));
	end
	
	% Saving models
	if(saveModel)
		save(['data/GMM_ME_PD' letter '.mat'], 'modelPD');
		save(['data/GMM_ME_x' letter '.mat'], 'modelKin');
	end
	

	%% Plots
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	figure('position',[10 10 1000 450],'color',[1 1 1]);
	clrmap = lines(nbSamples);
	% Plot demonstrations of velocity manipulability ellipsoids
	subplot(1,2,1); hold on;
	for n=1:nbSamples
		for t=round(linspace(1,nbData,15))
			plotGMM([s(n).Data(1,t);s(n).Data(2,t)], 1E-1*X(2:3,2:3,t+(n-1)*nbData), clrmap(n,:), .4); % Scaled matrix!
		end
	end
	for n=1:nbSamples % Plots 2D Cartesian trajectories
		plot(s(n).Data(1,:),s(n).Data(2,:), 'color', [0.6 0.6 0.6], 'Linewidth', 2); % Scaled matrix!
	end
	axis equal;
	xlabel('x_1'); ylabel('x_2');
	% Plot mean of Gaussian components for GMM encoding manipulabilities
	clrmap = lines(modelPD.nbStates);
	subplot(1,2,2); hold on;
	for i=1:modelPD.nbStates
		plotGMM(zeros(2,1), modelPD.MuMan(2:end,2:end,i), clrmap(i,:), .3);
	end
	axis equal;
	xlabel('x_1'); ylabel('x_2');
	
	% Plot demonstrations of velocity manipulability ellipsoids over time
	figure('position',[10 10 1000 450],'color',[1 1 1]);
	clrmap = lines(nbSamples);
	for n=1:nbSamples
		subplot(3,2,n); hold on;
		for t=round(linspace(1,nbData,15))
			plotGMM([t;0], X(2:3,2:3,t+(n-1)*nbData), clrmap(n,:), .4); % Scaled matrix!
		end
		xaxis(-10, nbData+10);
		xlabel('t');
	end
	
	subplot(3,2,nbSamples+1); hold on;
	for n=1:nbSamples
		for t=round(linspace(1,nbData,15))
			plotGMM([t;0], X(2:3,2:3,t+(n-1)*nbData), clrmap(n,:), .4); % Scaled matrix!
		end
	end
	xaxis(-10, nbData+10);
	xlabel('t');
	
	drawnow;
else
	load(['data/GMM_ME_PD' letter '.mat']);
	load(['data/GMM_ME_x' letter '.mat']);
end


%% GMR (version with single optimization loop)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Regression...');
xIn = zeros(1,nbData);
xIn(1,:) = [1:nbData] * modelPD.dt;

in=1; % Input dimension
out=2:modelPD.nbVar; % Output dimensions for GMM over manipulabilities
nbVarOut = length(out);

uhat = zeros(nbVarOut,nbVarOut,nbData);
xhat = zeros(nbVarOut,nbVarOut,nbData);
uOutTmp = zeros(nbVarOut,nbVarOut,modelPD.nbStates,nbData);
SigmaTmp = zeros(modelPD.nbVarCovOut,modelPD.nbVarCovOut,modelPD.nbStates);
expSigma = zeros(nbVarOut,nbVarOut,nbVarOut,nbVarOut,nbData);
H = [];

% Initial conditions for manipulability transfer and robot control
% q0s = [pi/4 pi/4 pi/9 pi/6 pi/2.5]'; % Initial robot configuration
q0s = [pi/4 pi/4 pi/2 -pi/18 -pi/2]'; % Initial robot configuration
qt = q0s;
rbtS.pos = [];

figure('position',[10 10 1000 450],'color',[1 1 1]);
for t=1:nbData
	% GMR for 2D Cartesian trajectory
	[xd, ~] = GMR(modelKin, t*modelKin.dt, in, 2:modelKin.nbVar);
	
	% GMR for manipulability ellipsoids
	% Compute activation weight
	for i=1:modelPD.nbStates
		H(i,t) = modelPD.Priors(i) * gaussPDF(xIn(:,t)-modelPD.MuMan(in,in,i),...
			modelPD.Mu(in,in,i), modelPD.Sigma(in,in,in,in,i));
	end
	H(:,t) = H(:,t) / sum(H(:,t)+realmin);
	
	% Compute conditional mean (with covariance transportation)
	if t==1
		[~,id] = max(H(:,t));
		xhat(:,:,t) = modelPD.MuMan(out,out,id); % Initial point
	else
		xhat(:,:,t) = xhat(:,:,t-1);
	end
	for n=1:nbIter
		uhat(:,:,t) = zeros(nbVarOut,nbVarOut);
		for i=1:modelPD.nbStates
			% Transportation of covariance from model.MuMan(outMan,i) to xhat(:,t)
			S1 = modelPD.MuMan(out,out,i);
			S2 = xhat(:,:,t);
			Ac = blkdiag(1,transp(S1,S2));
			
			% Parallel transport of eigenvectors
			for j = 1:size(modelPD.V,3)
				pV(:,:,j,i) = Ac * modelPD.D(j,j,i)^.5 * modelPD.V(:,:,j,i) * Ac';
			end
			
			% Parallel transported sigma (reconstruction from eigenvectors)
			pSigma(:,:,:,:,i) = zeros(size(modelPD.Sigma(:,:,:,:,i)));
			for j = 1:size(modelPD.V,3)
				pSigma(:,:,:,:,i) = pSigma(:,:,:,:,i) + outerprod(pV(:,:,j,i),pV(:,:,j,i));
			end
			
			[SigmaTmp(:,:,i), pV(:,:,:,i), pD(:,:,i)] = covOrder4to2(pSigma(:,:,:,:,i));
			
			% Gaussian conditioning on the tangent space
			uOutTmp(:,:,i,t) = logmap(modelPD.MuMan(out,out,i), xhat(:,:,t)) + ...
				tensor4o_mult(tensor4o_div(pSigma(out,out,in,in,i),pSigma(in,in,in,in,i)), (xIn(:,t)-modelPD.MuMan(in,in,i)));
			
			uhat(:,:,t) = uhat(:,:,t) + uOutTmp(:,:,i,t) * H(i,t);
		end
		
		xhat(:,:,t) = expmap(uhat(:,:,t), xhat(:,:,t));
	end
	
	% Compute conditional covariances
	for i=1:modelPD.nbStates
		expSigma(:,:,:,:,t) = expSigma(:,:,:,:,t) + H(i,t) * (pSigma(out,out,out,out,i) ...
			- tensor4o_mult(tensor4o_div(pSigma(out,out,in,in,i),pSigma(in,in,in,in,i)),pSigma(in,in,out,out,i)));
	end
	
	% Robot control
	% Redundancy resolution for desired manipulability ellipsoid
	Cgrad = ManipulabilityCostGradient(typeCost, ME_c, xhat(:,:,t), qSym);
	for n = 1 : nbIterME
		Jt = robotS.jacob0(qt); % Current Jacobian
		Jt = Jt(1:2,:);
		Htmp = robotS.fkine(qt); % Forward Kinematics
		
		% Compatibility with 9.X and 10.X versions of robotics toolbox
		if isobject(Htmp) % SE3 object verification
			xt = Htmp.t(1:2);
		else
			xt = Htmp(1:2,end);
		end
		
		rbtS.pos = [rbtS.pos xt];
		
		% Evaluating cost gradient for current joint configuration
		Cgrad_t = Cgrad(qt(1),qt(2),qt(3),qt(4),qt(5));
		
		% Desired joint velocities
		dq_T1 = pinv(Jt)*(modelPD.Kp*(xd - xt));	% Main qtask joint velocities
		dq_ns = -(eye(robotS.n) - pinv(Jt)*Jt) * alpha * Cgrad_t; % Redundancy resolution
		
		% Updating joint position
		qt = qt + (dq_T1 + dq_ns)*modelPD.dt;
	end
	
	% Plotting robot and VMEs
	if(mod(t,10) == 0)
		Jt = robotS.jacob0(qt); % Current Jacobian
		Jt = Jt(1:2,:);
		
		subplot(1,3,1); hold on; % Desired and actual manipulability ellipsoids
		plotGMM(xt, 1E-2*xhat(:,:,t), [0.2 0.8 0.2], .4); % Scaled matrix!
		plotGMM(xt, 1E-2*(Jt*Jt'), [0.8 0.2 0.2], .4); % Scaled matrix!
		colTmp = [1,1,1] - [.7,.7,.7] * t/nbData;
		plot(rbtS.pos(1,:),  rbtS.pos(2,:), 'color', [0.8 0.8 0.8], 'Linewidth', 1);
		axis square;
		axis equal;
		xlabel('x_1'); ylabel('x_2');
		
		subplot (3,2,2); hold on; % Desired and actual manipulability ellipsoids over time
		plotGMM([t;0], xhat(:,:,t), [0.2 0.8 0.2], .4); % Scaled matrix!
		plotGMM([t;0], Jt*Jt', [0.8 0.2 0.2], .4); % Scaled matrix!
		xaxis(-10, nbData+10);
		drawnow;
	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 8 8],'position',[10,10,1350,650]);
clrmap = lines(modelPD.nbStates);

subplot(3,1,1); hold on; %axis off;
sc = 1/modelPD.dt;
if(needsLearning)
	for t=1:size(X,3) % Plotting man. ellipsoids from demonstration data
		plotGMM([X(in,in,t)*sc; 0], X(out,out,t), [.6 .6 .6], .1);
	end
end
for i=1:modelPD.nbStates % Plotting GMM of man. ellipsoids
	plotGMM([modelPD.MuMan(in,in,i)*sc; 0], modelPD.MuMan(out,out,i), clrmap(i,:), .3);
end
xaxis(xIn(1)*sc, xIn(end)*sc);

subplot(3,1,2); hold on;
for t=1:nbData % Plotting estimated man. ellipsoid from GMR
	plotGMM([xIn(1,t)*sc; 0], xhat(:,:,t), [0 1 0], .1);
end
xaxis(xIn(1)*sc, xIn(end)*sc);

subplot(3,1,3); hold on;
for i=1:modelPD.nbStates % Plotting states influence during GMR estimation
	plot(xIn, H(i,:),'linewidth',2,'color',clrmap(i,:));
end
axis([xIn(1), xIn(end), 0, 1.02]);
set(gca,'xtick',[],'ytick',[]);
xlabel('t'); ylabel('h_i');

end


%%%%%%%%%%%%%%%%%%
function X = expmap(U,S)
	% Exponential map (SPD manifold)
	N = size(U,3);
	for n = 1:N
		X(:,:,n) = S^.5 * expm(S^-.5 * U(:,:,n) * S^-.5) * S^.5;
	end
end

%%%%%%%%%%%%%%%%%%
function U = logmap(X,S)
	% Logarithm map (SPD manifold)
	N = size(X,3);
	for n = 1:N
		U(:,:,n) = S^.5 * logm(S^-.5 * X(:,:,n) * S^-.5) * S^.5;
	end
end

%%%%%%%%%%%%%%%%%%
function Ac = transp(S1,S2)
	% Parallel transport (SPD manifold)
	t = 1;
	U = logmap(S2,S1);
	Ac = S1^.5 * expm(0.5 .* t .* S1^-.5 * U * S1^-.5) * S1^-.5;
	%Ac2 = S1^-.5 * expm(0.5 .* t .* S1^-.5 * U * S1^-.5) * S1^.5;
	%Transportation operator to move V from S1 to S2: VT = Ac*V*Ac2 = Ac*V*Ac'?
	%Computationally economic way: Ac = (S2/S1)^.5
end

%%%%%%%%%%%%%%%%%%
function M = spdMean(setS, nbIt)
	% Mean of SPD matrices on the manifold
	if nargin == 1
		nbIt = 10;
	end
	M = setS(:,:,1);

	for i=1:nbIt
		L = zeros(size(setS,1),size(setS,2));
		for n = 1:size(setS,3)
			L = L + logm(M^-.5 * setS(:,:,n)* M^-.5);
		end
		M = M^.5 * expm(L./size(setS,3)) * M^.5;
	end
end

%%%%%%%%%%%%%%%%%%
function model = spd_init_GMM_kbins(Data, model, nbSamples)
	% K-Bins initialisation by relying on SPD manifold
	% Parameters
	nbData = size(Data,3) / nbSamples;
	if ~isfield(model,'params_diagRegFact')
		model.params_diagRegFact = 1E-4; %Optional regularization term to avoid numerical instability
	end
	e0tensor = zeros(model.nbVar, model.nbVar, model.nbVar, model.nbVar);
	for i = 1:model.nbVar
		e0tensor(i,i,i,i) = 1;
	end
	% Delimit the cluster bins for the first demonstration
	tSep = round(linspace(0, nbData, model.nbStates+1));

	% Compute statistics for each bin
	for i=1:model.nbStates
		id=[];
		for n=1:nbSamples
			id = [id (n-1)*nbData+[tSep(i)+1:tSep(i+1)]];
		end
		model.Priors(i) = length(id);
		model.MuMan(:,:,i) = spdMean(Data(:,:,id));

		DataTgt = zeros(size(Data(:,:,id)));
		for n = 1:length(id)
			DataTgt(:,:,n) = logmap(Data(:,:,id(n)),model.MuMan(:,:,i));
		end
		model.Sigma(:,:,:,:,i) = computeCov(DataTgt) + e0tensor.*model.params_diagRegFact;

	end
	model.Priors = model.Priors / sum(model.Priors);
end

function M = tensor2mat(T, rows, cols)
	% Matricisation of a tensor
	% The rows, respectively columns of the matrix are 'rows', respectively 'cols' of the tensor.
	if nargin <=2
		cols = [];
	end

	sizeT = size(T);
	N = ndims(T);

	if isempty(rows)
		rows = 1:N;
		rows(cols) = [];
	end
	if isempty(cols)
		cols = 1:N;
		cols(rows) = [];
	end

	M = reshape(T,prod(sizeT(rows)), prod(sizeT(cols)));
end

%%%%%%%%%%%%%%%%%%
function T = tensor4o_mult(A,B)
	% Multiplication of two 4th-order tensors A and B
	if ndims(A) == 4 || ndims(B) == 4;
		sizeA = size(A);
		sizeB = size(B);
		if ismatrix(A)
			sizeA(3:4) = [1,1];
		end
		if ismatrix(B)
			sizeB(3:4) = [1,1];
		end

		if sizeA(3) ~= sizeB(1) || sizeA(4) ~= sizeB(2)
			error('Dimensions mismatch: two last dim of A should be the same than two first dim of B.');
		end

		T = zeros(sizeA(1),sizeA(2),sizeB(3),sizeB(4));

		for i = 1:sizeA(3)
			for j = 1:sizeA(4)
				T = T + outerprod(A(:,:,i,j),permute(B(i,j,:,:),[3,4,1,2]));
			end
		end
	else
		if ismatrix(A) && isscalar(B)
			T = A*B;
		else
			error('Dimensions mismatch.');
		end
	end
end

%%%%%%%%%%%%%%%%%%
function T = tensor4o_div(A,B)
	% Division of two 4th-order tensors A and B
	if ndims(A) == 4 || ndims(B) == 4;
		sizeA = size(A);
		sizeB = size(B);
		if ismatrix(A)
			sizeA(3:4) = [1,1];
		end
		if ismatrix(B)
			T = A/B;
		else
			if sizeA(3) ~= sizeB(1) || sizeA(4) ~= sizeB(2)
				error('Dimensions mismatch: two last dim of A should be the same than two first dim of B.');
			end

			[~, V, D] = covOrder4to2(B);
			invB = zeros(size(B));
			for j = 1:size(V,3)
				invB = invB + D(j,j)^-1 .* outerprod(V(:,:,j),V(:,:,j));
			end

			T = zeros(sizeA(1),sizeA(2),sizeB(3),sizeB(4));

			for i = 1:sizeA(3)
				for j = 1:sizeA(4)
					T = T + outerprod(A(:,:,i,j),permute(invB(i,j,:,:),[3,4,1,2]));
				end
			end
		end
	else
		if ismatrix(A) && isscalar(B)
			T = A/B;
		else
			error('Dimensions mismatch.');
		end
	end
end

%%%%%%%%%%%%%%%%%%
function prob = gaussPDF2(Data, Mu, Sigma)
	% Likelihood of datapoint(s) to be generated by a Gaussian parameterized by
	% center and covariance. The inverse and determinant of the covariance are
	% computed using the eigenvalue decomposition.
	[nbVar,nbData] = size(Data);
	Data = Data' - repmat(Mu',nbData,1);
	[V,D] = eig(Sigma);
	SigmaInv = V*diag(diag(D).^-1)*V';
	prob = sum((Data*SigmaInv).*Data, 2);
	prob = exp(-0.5*prob) / sqrt((2*pi)^nbVar * abs(det(Sigma)) + realmin);
end

%%%%%%%%%%%%%%%%%%
function S = computeCov(Data)
	% Compute the 4th-order covariance of matrix data.
	d = size(Data,1);
	N = size(Data,3);

	Data = Data-repmat(mean(Data,3),1,1,N);

	S = zeros(d,d,d,d);
	for i = 1:N
		S = S + outerprod(Data(:,:,i),Data(:,:,i));
	end
	S = S./(N-1);
end

%%%%%%%%%%%%%%%%%%
function v = symMat2Vec(S)
	% Reduced vectorisation of a symmetric matrix.
	[d,~,N] = size(S);

	v = zeros(d+d*(d-1)/2,N);
	for n = 1:N
		v(1:d,n) = diag(S(:,:,n));

		row = d+1;
		for i = 1:d-1
			v(row:row + d-1-i,n) = sqrt(2).*S(i+1:end,i,n);
			row = row + d-i;
		end
	end
end

%%%%%%%%%%%%%%%%%%
function id = cov_findNonZeroID(in, out, isVec_in, isVec_out)
	if nargin == 2
		isVec_in = 0;
		isVec_out = 0;
	end

	numberMat = zeros(out(end));
	if ~isVec_in
		numberMat(in,in) = ones(length(in));
	else
		numberMat(in,in) = eye(length(in));
	end
	if ~isVec_out
		numberMat(out,out) = ones(length(out));
	else
		numberMat(out,out) = eye(length(out));
	end

	numberVec = logical(symMat2Vec(numberMat))';
	numberVec = numberVec.*[1:length(numberVec)];

	id = nonzeros(numberVec)';
end

%%%%%%%%%%%%%%%%%%
function [covOrder4to2, covOrder2to4] = set_covOrder4to2(dim)
	% Set the factors necessary to simplify a 4th-order covariance of symmetric
	% matrix to a 2nd-order covariance. The dimension ofthe 4th-order
	% covariance is dim x dim x dim x dim.Return the functions covOrder4to2 and
	% covOrder2to4. This function must be called one time for each
	% covariance's size.
	newDim = dim+dim*(dim-1)/2;

	% Computation of the indices and coefficients to transform 4th-order
	% covariances to 2nd-order covariances
	sizeS = [dim,dim,dim,dim];
	sizeSred = [newDim,newDim];
	id = [];
	idred = [];
	coeffs = [];

	% left-up part
	for k = 1:dim
		for m = 1:dim
			id = [id,sub2ind(sizeS,k,k,m,m)];
			idred = [idred,sub2ind(sizeSred,k,m)];
			coeffs = [coeffs,1];
		end
	end

	% right-down part
	row = dim+1; col = dim+1;
	for k = 1:dim-1
		for m = k+1:dim
			for p = 1:dim-1
				for q = p+1:dim
					id = [id,sub2ind(sizeS,k,m,p,q)];
					idred = [idred,sub2ind(sizeSred,row,col)];
					coeffs = [coeffs,2];
					col = col+1;
				end
			end
			row = row + 1;
			col = dim+1;
		end
	end

	% side-parts
	for k = 1:dim
		col = dim+1;
		for p = 1:dim-1
			for q = p+1:dim
				id = [id,sub2ind(sizeS,k,k,p,q)];
				idred = [idred,sub2ind(sizeSred,k,col)];
				id = [id,sub2ind(sizeS,k,k,p,q)];
				idred = [idred,sub2ind(sizeSred,col,k)];
				coeffs = [coeffs,sqrt(2),sqrt(2)];
				col = col+1;
			end
		end
	end

	% Computation of the indices and coefficients to transform eigenvectors to eigentensors
	sizeV = [dim,dim,newDim];
	sizeVred = [newDim,newDim];
	idEig = [];
	idredEig = [];
	coeffsEig = [];

	for n = 1:newDim
		% diagonal part
		for j = 1:dim
			idEig = [idEig,sub2ind(sizeV,j,j,n)];
			idredEig = [idredEig,sub2ind(sizeVred,j,n)];
			coeffsEig = [coeffsEig,1];
		end

		% side parts
		j = dim+1;
		for k = 1:dim-1
			for m = k+1:dim
				idEig = [idEig,sub2ind(sizeV,k,m,n)];
				idredEig = [idredEig,sub2ind(sizeVred,j,n)];
				idEig = [idEig,sub2ind(sizeV,m,k,n)];
				idredEig = [idredEig,sub2ind(sizeVred,j,n)];
				coeffsEig = [coeffsEig,1/sqrt(2),1/sqrt(2)];
				j = j+1;
			end
		end
	end

	function [Sred, V, D] = def_covOrder4to2(S)
		Sred = zeros(newDim,newDim);
		Sred(idred) = bsxfun(@times,S(id),coeffs);
		[v,D] = eig(Sred);
		V = zeros(dim,dim,newDim);
		V(idEig) = bsxfun(@times,v(idredEig),coeffsEig);
	end

	function [S, V, D] = def_covOrder2to4(Sred)
		[v,D] = eig(Sred);
		V = zeros(dim,dim,newDim);
		V(idEig) = bsxfun(@times,v(idredEig),coeffsEig);
		
		S = zeros(dim,dim,dim,dim);
		for i = 1:size(V,3)
			S = S + D(i,i).*outerprod(V(:,:,i),V(:,:,i));
		end
	end

	covOrder4to2 = @def_covOrder4to2;
	covOrder2to4 = @def_covOrder2to4;
end