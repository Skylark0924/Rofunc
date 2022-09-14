function demo_HSMM_adaptiveDuration_infHor01
% Hidden semi-Markov model with adaptive duration, used with a controller  
% based on discrete infinite-horizon LQR (with position and velocity tracking).
%
% - This model adapts the log-normal state duration probability according
%		to an external input "u".
% - Every model state has a duration probability represented by a mixture of
%   Gaussians.
% - A conditional Gaussian distribution is obtained at each time step by
%   applying GMR given the external input "u".
%
% This code:
%		1. Sets the variable values for the ADHSMM, and a linear quadratic
%		   regulator in charge of following a step-wise trajectory obtained
%			 from the forward variable of the model.
%		2. Loads synthetic data to be used for training the model. The data
%			 correspond to several G-shape trajectories in 2D.
%		3. Trains the model in two phases: (i) an HSMM is trained, (ii)
%			 duration probabilities (GMM) for each model state are set manually
%			 (these can be easily learned from EM for a Gaussian mixture model).
%		4. Reconstructs a state sequence where the state duration depends on
%			 the given external input.
%		5. Retrieves a reproduction trajectory by implementing a linear
%			 quadratic regulator that follows the step-wise reference obtained
%			 from the state sequence previously computed.
%		6. Plots the results as animated graphs.
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
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
% Written by Leonel Rozo and Sylvain Calinon, 2019
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbSamples = 5; %Number of demonstrations
nbData = 100; %Number of datapoints in a trajectory
onlyDur = 1; %Forward variable parameter (0 for standard HSMM computation, 1 for HSMM considering only duration)

model.nbStates = 6; %Number of states
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.dt = 0.01; %Time step duration
model.minSigmaPd = 1E-3; %Minimum variance of state duration (regularization term)
model.rfactor = 1E-4;	%Control cost in LQR (to be set carefully because infinite horizon LQR can suffer mumerical instability)

%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;

%Artificial trigerring of external input
%u = zeros(nbData,1); %No perturbation signal
u = [zeros(23,1); ones(22,1); zeros(55,1)]; %Simulation of perturbation signal


%% Discrete dynamical System settings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %Integration with Euler method 
% Ac1d = diag(ones(model.nbDeriv-1,1),1); %Continuous 1D
% Bc1d = [zeros(model.nbDeriv-1,1); 1]; %Continuous 1D
% A = kron(eye(model.nbDeriv)+Ac1d*model.dt, eye(model.nbVarPos)); %Discrete
% B = kron(Bc1d*model.dt, eye(model.nbVarPos)); %Discrete

%Integration with higher order Taylor series expansion
A1d = zeros(model.nbDeriv);
for i=0:model.nbDeriv-1
	A1d = A1d + diag(ones(model.nbDeriv-i,1),i) * model.dt^i * 1/factorial(i); %Discrete 1D
end
B1d = zeros(model.nbDeriv,1); 
for i=1:model.nbDeriv
	B1d(model.nbDeriv-i+1) = model.dt^i * 1/factorial(i); %Discrete 1D
end
A = kron(A1d, eye(model.nbVarPos)); %Discrete nD
B = kron(B1d, eye(model.nbVarPos)); %Discrete nD

% %Conversion with control toolbox
% Ac1d = diag(ones(model.nbDeriv-1,1),1); %Continuous 1D
% Bc1d = [zeros(model.nbDeriv-1,1); 1]; %Continuous 1D
% Cc1d = [1, zeros(1,model.nbDeriv-1)]; %Continuous 1D
% sysd = c2d(ss(Ac1d,Bc1d,Cc1d,0), model.dt); 
% A = kron(sysd.a, eye(model.nbVarPos));
% B = kron(sysd.b, eye(model.nbVarPos));


%% Load handwriting dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data/2Dletters/S.mat');
Data=[];
for n=1:nbSamples
	s(n).Data=[];
	for m=1:model.nbDeriv
		if m==1
			dTmp = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
		else
			dTmp = gradient(dTmp) / model.dt; %Compute derivatives
		end
		s(n).Data = [s(n).Data; dTmp];
	end
	Data0(:,n) = s(n).Data(:,1);
	Data = [Data s(n).Data]; 
end
Data0 = mean(Data0,2);


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Learning');
%model = init_GMM_kmeans(Data, model);
model = init_GMM_kbins(Data, model, nbSamples);

% %Random initialization
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
[model,H] = EM_HMM(s, model);

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
	%Second Gaussian: Slow down the movement by a factor 2 if the input u is 1 
	model.gmm_Pd(i).Mu(:,2) = [1; log(exp(mean(st(i).d))*2)]; 
	model.gmm_Pd(i).Sigma(:,:,2) = diag([1E-2, cov(st(i).d)+model.minSigmaPd]);
end


%% Reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
fprintf('Reproduction');
nbPd = round(3 * nbData/model.nbStates); %Number of maximum duration step to consider in the HSMM (3 is a safety factor)
%Initialization
r(1).Data = zeros(model.nbVar,nbData);	%Reproduction data
h = zeros(model.nbStates,nbData);
qList = zeros(nbData,1); %component id list
c = zeros(nbData,1); %scaling factor to avoid numerical issues
c(1) = 1; %Initialization of scaling factor
X = Data0;	%Initial state vector
%X = [Data0(1:model.nbVarPos,1)+randn(model.nbVarPos,1)*1E0; zeros(model.nbVar-model.nbVarPos,1)];	%Initial state vector

for t=1:nbData
	if mod(t,10)==1
		fprintf('.');
	end
	r(1).Data(:,t) = X; %Log position
	for i=1:model.nbStates
		% Conditional Gaussian distribution given the external input "u"
		[model.Mu_Pd(:,i), model.Sigma_Pd(:,:,i)] = GMR(model.gmm_Pd(i), u(t), 1, 2);
		% Pre-computation of duration probabilities
		model.Pd(i,:) = gaussPDF(log(1:nbPd), model.Mu_Pd(:,i), model.Sigma_Pd(:,:,i)) + realmin;
		% The rescaling formula below can be used to guarantee that the cumulated sum is one (to avoid numerical issues)
		model.Pd(i,:) = model.Pd(i,:) / sum(model.Pd(i,:));

		% HSMM forward variable
		if t <= nbPd
			if(onlyDur)
				oTmp = 1; %Observation probability for "duration-only HSMM"
			else
				oTmp = prod(c(1:t) .* gaussPDF(r(1).Data(:,1:t), model.Mu(:,i), model.Sigma(:,:,i))); %Observation probability for standard HSMM
				%oTmp = prod(c(1:t) .* gaussPDF(r(1).Data(1:model.nbVarPos,1:t), model.Mu(1:model.nbVarPos,i), model.Sigma(1:model.nbVarPos,1:model.nbVarPos,i))); %Observation probability for standard HSMM
			end
			h(i,t) = model.StatesPriors(i) * model.Pd(i,t) * oTmp;
		end
		for d=1:min(t-1,nbPd)
			if(onlyDur)
				oTmp = 1; %Observation probability for "duration-only HSMM"
			else
				oTmp = prod(c(t-d+1:t) .* gaussPDF(r(1).Data(:,t-d+1:t), model.Mu(:,i), model.Sigma(:,:,i))); %Observation prob. for HSMM
				%oTmp = prod(c(t-d+1:t) .* gaussPDF(r(1).Data(1:model.nbVarPos,t-d+1:t), model.Mu(1:model.nbVarPos,i), model.Sigma(1:model.nbVarPos,1:model.nbVarPos,i))); %Observation prob. for HSMM
			end
			h(i,t) = h(i,t) + h(:,t-d)' * model.Trans(:,i) * model.Pd(i,d) * oTmp;
		end
	end
	c(t+1) = 1/sum(h(:,t)+realmin); %Update of scaling factor

	% LQR tracking of stepwise reference
	[~,qList(t)] = max(h(:,t),[],1); %Get current component id
	Q = inv(model.Sigma(:,:,qList(t))); %Tracking cost
	P = solveAlgebraicRiccati_eig_discrete(A, B*(R\B'), (Q+Q')/2); %ARE for discrete systems
	L = (B'*P*B + R) \ B'*P*A; %Feedback gain (discrete version)
	DDX =  -L * (X - model.Mu(:,qList(t))); %Compute acceleration (with only feedback terms)
	
	% Emulating that the system stays at the same position when perturbed (if the external input u is equal to 1)
	if u(t)~=1
		X = A*X + B*DDX; %Update position
	end

	r(1).xTar(:,t) = model.Mu(:,qList(t)); %Log reference
	r(1).ddx(:,t) = DDX; %Log acceleration computed from LQR
	r(1).Pd(:,:,t) = model.Pd; %Log temporary state duration probabilities
end
h = h ./ repmat(sum(h,1),model.nbStates,1);
fprintf('\n');


%% Animated plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hf = figure('PaperPosition',[0 0 18 4.5],'position',[10,10,1300,450],'color',[1 1 1]);
clrmap = lines(model.nbStates);

%Spatial plot of the data
subplot(3,4,[1,5,9]); hold on;
for i=1:model.nbStates
	plotGMM(model.Mu(1:2,i), model.Sigma(1:2,1:2,i), clrmap(i,:), .6);
end
plot(Data(1,:), Data(2,:), '.', 'color', [.6 .6 .6]);
plot(r(1).Data(1,:), r(1).Data(2,:), '-','lineWidth', 2.5, 'color', [0.3 0.3 0.3]);
axis equal; axis([-10 10 -10 10]); axis off;

%Timeline plot of the duration probabilities
h1=[];
for t=1:1:nbData
	figure(hf); delete(h1);
	%Spatial plot of the data
	subplot(3,4,[1,5,9]); hold on;
	h1 = plot(r(1).xTar(1,t), r(1).xTar(2,t), 'x','lineWidth', 2.5, 'color', [0.2 0.9 0.2]);
	h1 = [h1 plot(r(1).Data(1,t), r(1).Data(2,t), 'o','lineWidth', 2.5, 'color', [0.9 0.6 0.3])];
	h1 = [h1 quiver(r(1).Data(1,t), r(1).Data(2,t), r(1).ddx(1,t)*1E-3, r(1).ddx(2,t)*1E-3, 25, 'LineWidth', 2, 'color', max(min(h(:,t)'*clrmap,1),0))];
	axis([-10 10 -10 10]); 

	subplot(3,4,2:4); hold on;
	for i=1:model.nbStates
		yTmp = r(1).Pd(i,:,t) / max(r(1).Pd(i,:,t));
		h1 = [h1 patch([1, 1:size(yTmp,2), size(yTmp,2)], [0, yTmp, 0], clrmap(i,:), 'EdgeColor', 'none', 'facealpha', .6)];
		h1 = [h1 plot(1:size(yTmp,2), yTmp, 'linewidth', 2, 'color', clrmap(i,:))];
	end
	axis([1 nbData 0 1]);
	ylabel('Pd','fontsize',16);

	%Timeline plot of the state sequence probabilities
	subplot(3,4,6:8); hold on;
	for i=1:model.nbStates
		h1 = [h1 patch([1, 1:t, t], [0, h(i,1:t), 0], clrmap(i,:), 'EdgeColor', 'none', 'facealpha', .6)];
		h1 = [h1 plot(1:t, h(i,1:t), 'linewidth', 2, 'color', clrmap(i,:))];
	end
	set(gca,'xtick',[10:10:nbData],'fontsize',8); axis([1 nbData -0.01 1.01]);
	ylabel('h','fontsize',16);

	subplot(3,4,10:12); hold on;
	h1 = [h1 patch([1 1:t t], [0 u(1:t)' 0], [1 .8 .8], 'LineWidth', 2, 'EdgeColor', [.8 0 0])];
	set(gca,'xtick',[10:10:nbData],'fontsize',8); axis([1 nbData -0.01 1.01]);
	xlabel('t','fontsize',16);
	ylabel('u','fontsize',16);
	
	drawnow;
	%pause(0.01);
end


%% Static plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 18 4.5],'position',[10,200,1300,450],'color',[1 1 1]);
for i=1:2
	limAxes = [1, nbData, min(Data(i,:))-1E0, max(Data(i,:))+1E0];
	subplot(1,2,i); hold on; box on;
	msh=[]; x0=[];
	patch(1:nbData, u*(limAxes(4)-limAxes(3))+limAxes(3), [1 .8 .8],'edgecolor',[1 .8 .8]);
	
	for t=1:nbData-1
		if size(msh,2)==0
			msh(:,1) = [t; model.Mu(i,qList(t))];
		end
		if t==nbData-1 || qList(t+1)~=qList(t)
			%Reference
			msh(:,2) = [t+1; model.Mu(i,qList(t))];
			%Variance
			sTmp = model.Sigma(i,i,qList(t))^.5;
			%Mesh for patch (variance)
			msh2 = [msh(:,1)+[0;sTmp], msh(:,2)+[0;sTmp], msh(:,2)-[0;sTmp], msh(:,1)-[0;sTmp], msh(:,1)+[0;sTmp]];
			patch(msh2(1,:), msh2(2,:), [.85 .85 .85],'edgecolor',[.7 .7 .7]);
			plot(msh(1,:), msh(2,:), '-','linewidth',3,'color',[.7 .7 .7]);
			plot([msh(1,1) msh(1,1)], limAxes(3:4), ':','linewidth',1,'color',[.7 .7 .7]);
			x0 = [x0 msh];
			msh=[];
		end
	end
	plot(r(1).xTar(i,:), '.','lineWidth', 2.5, 'color', [0.5 0.5 0.5]);
	plot(r(1).Data(i,:), '-','lineWidth', 2.5, 'color', [0.3 0.3 0.3]);
	xlabel('t','fontsize',18);
	ylabel(['x_' num2str(i)],'fontsize',18);
	axis(limAxes);
end

pause;
close all;