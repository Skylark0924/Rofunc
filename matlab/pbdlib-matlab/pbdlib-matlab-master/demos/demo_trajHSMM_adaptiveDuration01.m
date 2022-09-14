function demo_trajHSMM_adaptiveDuration01
% Trajectory-HSMM with adaptive duration, by using an LQR controller for online trajectory retrieval.
%
% - A time window is defined in order to carry out the optimization process
%		of the trajectory GMM for a specific number of steps, given a state
%		sequence.
% - The ADHSMM model adapts the log-normal state duration probability
%		according to an external input "u".
% - Every ADHSMM state has a duration probability represented by a mixture
%		of Gaussians.
% - A conditional Gaussian distribution is obtained at each time step by
%   applying GMR given the external input "u".
% - Infinite LQR (with position and velocity references) is used, under the
%   assumption that there is not a specific finite horizon for the task.
%
% This code:
%		1. Sets the variable values for the ADHSMM , the trajectory retrieval
%			 model (trajGMM), and the linear quadratic regulator (LQR).
%		2. Loads synthetic data to be used for training the model. The data
%			 correspond to several G-shape trajectories in 2D.
%		3. Trains the model in two phases: (i) an HSMM is trained, (ii)
%			 duration probabilities (GMM) for each model state are set manually
%			 (these can be easily learned from EM for a Gaussian mixture model).
%		4. Reconstructs, in an online fashion, a state sequence where the state
%			 duration depends on the given external input.
%		5. Retrieves a reference trajectory from a weighted least squares
%		   approach that uses the means and covariance matrices from the state
%			 sequence previously computed.
%		6. Plots the results in dynamic graphs.
%
% If this code is useful for your research, please cite the related publication:
% @article{Rozo16Frontiers,
%   author="Rozo, L. and Silv\'erio, J. and Calinon, S. and Caldwell, D. G.",
%   title="Learning Controllers for Reactive and Proactive Behaviors in Human-Robot Collaboration",
%   journal="Frontiers in Robotics and {AI}",
%   year="2016",
%   month="June",
%   volume="3",
%   number="30",
%   pages="1--11",
%   doi="10.3389/frobt.2016.00030"
% }
% 
% Written by Leonel Rozo and Sylvain Calinon, 2016
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


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 6; % Number of components in the model
model.nbVarPos = 2; % Dimension of position data (here: x1,x2)
model.nbDeriv = 2; % Number of static&dynamic features (D=2 for [x,dx])
model.dt = 1; % Time step (large values such as 1 will tend to create clusers by following position information)
model.nbVar = model.nbVarPos * model.nbDeriv;
model.minSigmaPd = 1E-3; %Minimum variance of state duration (regularization term)

nbSamples = 5; % Number of trajectory samples
nbData = 200; % Number of datapoints in a trajectory
ctrdWndw = 1;   % Uses a centered window for the online implementation
Tw = 50;  % Time length for centered window implementation
onlyDur	= 0; %Forward variable parameter (0 for standard HSMM computation, 1 for HSMM considering only duration)
rFactor = 2E1; % R factor for infinite LQR

% Construct operator PHI (big sparse matrix)
[~,PHI1,PHI0] = constructPHI(model, nbData, nbSamples);

%Artificial trigerring of external input
u = zeros(nbData,1); %No perturbation signal
% u = [zeros(45,1); ones(45,1); zeros(110,1)]; %Simulation of perturbation signal


%% Load handwriting dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data/2Dletters/S.mat');
Data = [];
for n=1:nbSamples
	% Resampling
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData));

	% Re-arrange data in vector form for computing derivatives
	x = reshape(s(n).Data, numel(s(n).Data), 1);
	zeta = PHI1 * x;
	dataTmp = reshape(zeta, model.nbVarPos*model.nbDeriv, nbData);
	s(n).Data = dataTmp;
	Data0(:,n) = s(n).Data(:,1);
	Data = [Data dataTmp];
end
Data0 = mean(Data0,2);


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Learning');
% model = init_GMM_kmeans(Data, model);
model = init_GMM_kbins(Data, model, nbSamples);

% -----> Transition matrix initialization
% Random initialization for transition matrix
% model.Trans = rand(model.nbStates,model.nbStates);
% model.Trans = model.Trans ./ repmat(sum(model.Trans,2),1,model.nbStates);
% model.StatesPriors = rand(model.nbStates,1);
% model.StatesPriors = model.StatesPriors/sum(model.StatesPriors);

%Left-right model initialization
model.Trans = zeros(model.nbStates);
for i=1:model.nbStates-1
	model.Trans(i,i) = 1-(model.nbStates/nbData);
	model.Trans(i,i+1) = model.nbStates/nbData;
end
model.Trans(model.nbStates,model.nbStates) = 1.0;
model.StatesPriors = zeros(model.nbStates,1);
model.StatesPriors(1) = 1;
model.Priors = ones(model.nbStates,1);

%EM parameters learning
model.params_diagRegFact = 1E-3;
[model, H] = EM_HMM(s, model);

%Removal of self-transition (for HSMM representation) and normalization
model.Trans = model.Trans - diag(diag(model.Trans)) + eye(model.nbStates)*realmin;
model.Trans = model.Trans ./ repmat(sum(model.Trans,2),1,model.nbStates);

%Post-estimation of the state duration from data 
for i=1:model.nbStates
	st(i).d=[];
end
[~,hmax] = max(H);
currState = hmax(1);
cnt = 1;
for t=1:length(hmax)
	if (hmax(t)==currState)
		cnt = cnt+1;
	else
		st(currState).d = [st(currState).d log(cnt)];
		cnt = 1;
		currState = hmax(t);
	end
end
st(currState).d = [st(currState).d log(cnt)];

%Set state duration manually (as an example) with: u=0 -> normal duration, u=1 -> Twice longer duration
for i=1:model.nbStates
	model.gmm_Pd(i).nbStates = 2; %Two Gaussians composing the state duration probability
	model.gmm_Pd(i).Priors = ones(model.gmm_Pd(i).nbStates,1);
	%First Gaussian: normal behavior
	model.gmm_Pd(i).Mu(:,1) = [0; mean(st(i).d)]; 
	model.gmm_Pd(i).Sigma(:,:,1) = diag([1E-2, cov(st(i).d)+model.minSigmaPd]);
	%Second Gaussian: Slow down the movement by a factor 2 if the input is 1 
	model.gmm_Pd(i).Mu(:,2) = [1; mean(st(i).d)*2]; 
	model.gmm_Pd(i).Sigma(:,:,2) = diag([1E-2, 0.1]);
end


%% Reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Reproduction');
nbD = round(2 * nbData/model.nbStates); %Number of maximum duration step to consider in the HSMM (2 is a safety factor)
%Initialization
r(1).Data = zeros(model.nbVarPos,nbData);	%Reproduction data
h = zeros(model.nbStates,nbData);
c = zeros(nbData,1); %scaling factor to avoid numerical issues
c(1) = 1; %Initialization of scaling factor
X = Data0(1:model.nbVarPos,1); %Initial position vector
dX = zeros(model.nbVarPos,1);	%Initial velocity vector

% Compute PHI operator for current time window size
if(ctrdWndw)
	PHIw = kron(PHI0(1:(2*Tw+1)*model.nbDeriv, 1:(2*Tw+1)),eye(model.nbVarPos));
else
	PHIw = kron(PHI0(1:(2*Tw)*model.nbDeriv, 1:2*Tw), eye(model.nbVarPos));
end

for t=1:nbData
	if mod(t,10)==1
		fprintf('.');
	end
	r(1).Data(:,t) = X; %Log position
	for i=1:model.nbStates
		% Conditional Gaussian distribution given the external input "u"
		[model.Mu_Pd(:,i), model.Sigma_Pd(:,:,i), ~] = GMR(model.gmm_Pd(i),u(t), 1, 2);
		% Computation of duration probabilities
		model.Pd(i,:) = gaussPDF(log(1:nbD),model.Mu_Pd(:,i),model.Sigma_Pd(:,:,i)) + realmin;
		% The rescaling formula below can be used to guarantee that the cumulated sum is one (to avoid the numerical issues)
		model.Pd(i,:) = model.Pd(i,:) / sum(model.Pd(i,:));

		if t <= nbD
			if(onlyDur)
				oTmp = 1; %Observation probability for "duration-only HSMM"
			else
				%oTmp = prod(c(1:t) .* gaussPDF(r(1).Data(:,1:t), model.Mu(:,i), model.Sigma(:,:,i))'); %Observation probability for standard HSMM
				oTmp = prod(c(1:t) .* gaussPDF(r(1).Data(1:model.nbVarPos,1:t), model.Mu(1:model.nbVarPos,i), model.Sigma(1:model.nbVarPos,1:model.nbVarPos,i))'); % Observation probability for standard HSMM
			end
			h(i,t) = model.StatesPriors(i) * model.Pd(i,t) * oTmp;
		end
		for d=1:min(t-1,nbD)
			if(onlyDur)
				oTmp = 1; %Observation probability for "duration-only HSMM"
			else
				%oTmp = prod(c(t-d+1:t) .* gaussPDF(r(1).Data(:,t-d+1:t), model.Mu(:,i), model.Sigma(:,:,i))'); %Observation prob. for HSMM
				oTmp = prod(c(t-d+1:t) .* gaussPDF(r(1).Data(1:model.nbVarPos,t-d+1:t), model.Mu(1:model.nbVarPos,i), model.Sigma(1:model.nbVarPos,1:model.nbVarPos,i))'); %Observation probability for standard HSMM
			end
			h(i,t) = h(i,t) + h(:,t-d)' * model.Trans(:,i) * model.Pd(i,d) * oTmp;
		end
	end
	c(t+1) = 1 / sum(h(:,t)+realmin); %Update of scaling factor
	
	% Trajectory retrieval for time window
	if(ctrdWndw) % Centered time window [t-Tw,t+Tw]
		% Saving "Tw" weights based on previous observations
		H = zeros(model.nbStates, 2*Tw+1);
		cnt = 0;
		for tt = min(t, Tw+1) : -1 : 1
			H(:,tt) = h(:,t-cnt);
			cnt = cnt+1;
		end
		% Predict future weights (not influenced by position data)
		for tt=min(t, Tw+1)+1:2*Tw+1
			for i=1:model.nbStates
				for d=1:min(tt-1, nbD)
					H(i,tt) = H(i,tt) + H(:,tt-d)' * model.Trans(:,i) * model.Pd(i,d);
				end
			end
		end
	else % Time-window [t,2*Tw]
		% Predict future weights (not influenced by position data)
		H = zeros(model.nbStates, 2*Tw);
		H(:,1) = h(:,t);
		for tt = 2 : 2*Tw
			for i=1:model.nbStates
				for d=1:min(tt-1,nbD)
					H(i,tt) = H(i,tt) + H(:,tt-d)' * model.Trans(:,i) * model.Pd(i,d);
				end
			end
		end
	end
	H = H ./ repmat(sum(H,1),model.nbStates,1);
	
	% % Compute state path
	[~,qList] = max(H,[],1); %works also for nbStates=1
	% Concatenating mean vectors and covariance matrices
	MuQ = zeros(length(qList)*model.nbVar, 1);
	SigmaQ = zeros(length(qList)*model.nbVar, length(qList)*model.nbVar);
	for tt = 1 : length(qList)
		id1 = (tt-1)*model.nbVar+1:tt*model.nbVar;
		MuQ(id1,1) = model.Mu(:,qList(tt));
		SigmaQ(id1,id1) = model.Sigma(:,:,qList(tt));
	end

	% Reconstruction for the time window
	% Retrieval of data with weighted least squares solution
	[zeta, ~, ~, Scov] = lscov(PHIw, MuQ, SigmaQ, 'chol');

	if(ctrdWndw)
		r(1).s(t).Data = reshape(zeta, model.nbVarPos, 2*Tw+1);
		r(1).Data(:,t) = r(1).s(t).Data(:, min(t, Tw+1));

		% Rebuild covariance by reshaping Scov
		for tt = 1 : 2*Tw+1
			id = (tt-1)*model.nbVarPos+1 : tt*model.nbVarPos;
			r(1).s(t).expSigma(:,:,tt) = Scov(id,id) * (2*Tw+1);
		end
		r(1).expSigma(:,:,t) = r(1).s(t).expSigma(:,:,min(t, Tw+1));
	else
		r(1).s(t).Data = reshape(zeta, model.nbVarPos, 2*Tw);
		r(1).Data(:,t) = r(1).s(t).Data(:,1);

		% Rebuild covariance by reshaping Scov
		for tt = 1 : 2*Tw
			id = (tt-1)*model.nbVarPos+1 : tt*model.nbVarPos;
			r(1).s(t).expSigma(:,:,tt) = Scov(id,id) * (2*Tw);
		end
		r(1).expSigma(:,:,t) = r(1).s(t).expSigma(:,:,1);
	end

	% Using position and velocity data (NOT desired velocity)
	rr.currTar = [r(1).Data(:,t) ; zeros(model.nbVarPos,1)];
	rr.currSigma = [r(1).expSigma(:,:,t) zeros(model.nbVarPos,model.nbVarPos);...
		zeros(model.nbVarPos,model.nbVarPos) 100*eye(model.nbVarPos)];
	[rr,~] = reproduction_LQR_infiniteHorizon_withVel(model, rr, ...
		[X ; dX], rFactor);

	r(1).H(:,:,t) = H; %Log activation weight
	r(1).Pd(:,:,t) = model.Pd; %Log temporary state duration prob.
	
	if(~u(t)) % Emulating that the system stays at the same position when perturbed.
		X = rr.Data;
		dX = rr.dx;
	else
		dX = zeros(2,1);
	end
end
%toc
h = h ./ repmat(sum(h,1),model.nbStates,1);
fprintf('\n');


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[10,930,2300,900],'color',[1 1 1]);
% % Time series plot of the data
% for i=1:2
% 	limAxes = [1, nbData, min(Data(i,:))-1E0, max(Data(i,:))+1E0];
% 	subplot(1,2,i); hold on; box on;
% 	msh = [1:nbData, nbData:-1:1; ...
% 		r(1).Data(i,:)-squeeze(r(1).expSigma(i,i,:).^.5)'*1, ...
% 		fliplr(r(1).Data(i,:)+squeeze(r(1).expSigma(i,i,:).^.5)'*1)];
% 	patch(msh(1,:), msh(2,:), [.85 .85 .85],'edgecolor',[.7 .7 .7],...
% 		'edgealpha',.8,'facealpha',.5);
% 	plot(r(1).Data(i,:), '-','lineWidth', 2.5, 'color', [0.3 0.3 0.3]);
% 
% 	ylabel(['$x_' num2str(i) '$'],'interpreter','latex','fontsize',18);
% 	axis(limAxes);
% end

figure('position',[10,10,2300,900],'color',[1 1 1]);
clrmap = lines(model.nbStates);
%Spatial plot of the data
subplot(3,4,[1,5,9]); hold on; axis off;
for i=1:model.nbStates
	plotGMM(model.Mu(1:2,i), model.Sigma(1:2,1:2,i), clrmap(i,:), .6);
end
plot(Data(1,:), Data(2,:), '.', 'color', [.6 .6 .6]);
plot(r(1).Data(1,:), r(1).Data(2,:), '--','lineWidth', 2.5, 'color', [0.3 0.3 0.3]);
axis equal; axis([-10 10 -10 10]);  

%Timeline plot of the duration probabilities
h1=[];
for t=1:nbData
	delete(h1);
	%Spatial plot of the data
	subplot(3,4,[1,5,9]); hold on;
	h1 = plot(r(1).Data(1,t), r(1).Data(2,t), 'x','lineWidth', 2.5, 'color', [0.2 0.9 0.2]);
	h1 = [h1 plotGMM(r(1).Data(1:2,t), r(1).expSigma(:,:,t), [0.2 0.9 0.2], .5)];
	h1 = [h1 plot(r(1).s(t).Data(1,:), r(1).s(t).Data(2,:),'-','lineWidth',2.5,'color',[.8 0 0])];

	subplot(3,4,2:4); hold on;
	for i=1:model.nbStates
		yTmp = r(1).Pd(i,:,t) / max(r(1).Pd(i,:,t));
		h1 = [h1 patch([1, 1:size(yTmp,2), size(yTmp,2)], [0, yTmp, 0], clrmap(i,:), 'EdgeColor','none', 'facealpha', .6)];
		h1 = [h1 plot(1:size(yTmp,2), yTmp, 'linewidth', 2, 'color', clrmap(i,:))];
	end
	axis([1 nbData 0 1]);
	ylabel('Pd','fontsize',16);

	%Timeline plot of the state sequence probabilities
	subplot(3,4,6:8); hold on;
	for i=1:model.nbStates
		h1 = [h1 patch([1, 1:size(r(1).H,2), size(r(1).H,2)], [0, r(1).H(i,:,t), 0], clrmap(i,:), 'EdgeColor', 'none', 'facealpha', .6)];
		h1 = [h1 plot(1:size(r(1).H,2), r(1).H(i,:,t), 'linewidth', 2, 'color', clrmap(i,:))];
	end
	if(ctrdWndw)
		h1 = [h1 plot([Tw+1 Tw+1], [-1 1], '--', 'linewidth', 1, 'color', [0.5 0.5 0.5])];
	else
		h1 = [h1 plot([1 1], [-1 1], '--', 'linewidth', 1, 'color', [0.5 0.5 0.5])];
	end
	axis([1 size(r(1).H,2) -0.01 1.01]);
	ylabel('H','fontsize',16);

	subplot(3,4,10:12); hold on;
	for i=1:model.nbStates
		h1 = [h1 patch([1, 1:t, t], [0, h(i,1:t), 0], clrmap(i,:),'EdgeColor', 'none', 'facealpha', .6)];
		h1 = [h1 plot(1:t, h(i,1:t), 'linewidth', 2, 'color', clrmap(i,:))];
	end
	h1 = [h1 plot(1:t, u(1:t), 'linewidth', 2,'color', [0.5 0.5 0.5])];
	set(gca,'xtick',[10:10:nbData],'fontsize',8); axis([1 nbData -0.01 1.01]);
	xlabel('t','fontsize', 16);
	ylabel('h','fontsize', 16);
	drawnow;
end

pause;
close all;