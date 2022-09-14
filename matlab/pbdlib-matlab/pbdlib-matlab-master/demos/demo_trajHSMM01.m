function demo_trajHSMM01
% Trajectory synthesis with an HSMM with dynamic features (trajectory-HSMM).
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19chapter,
% 	author="Calinon, S. and Lee, D.",
% 	title="Learning Control",
% 	booktitle="Humanoid Robotics: a Reference",
% 	publisher="Springer",
% 	editor="Vadakkepat, P. and Goswami, A.", 
% 	year="2019",
% 	pages="1261--1312",
% 	doi="10.1007/978-94-007-6046-2_68"
% }
% 
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon, http://calinon.ch/
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


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 4; %Number of components in the GMM
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 3; %Number of static&dynamic features (D=2 for [x,dx], D=3 for [x,dx,ddx], etc.)
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.dt = 1; %Time step (without rescaling, large values such as 1 has the advantage of creating clusers based on position information)
nbSamples = 6; %Number of demonstrations
nbData = 100; %Number of datapoints in a trajectory
minSigmaPd = 1E1; %Minimum variance of state duration (regularization term)

[PHI,PHI1] = constructPHI(model,nbData,nbSamples); %Construct PHI operator (big sparse matrix)


%% Load handwriting movements
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data/2Dletters/S.mat');
Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	s(n).nbData = size(s(n).Data,2);
	Data = [Data s(n).Data]; 
end

%Re-arrange data in vector form
x = reshape(Data, model.nbVarPos*nbData*nbSamples, 1) * 1E2; %Scale data to avoid numerical computation problem
zeta = PHI*x; %zeta is for example [x1(1), x2(1), x1d(1), x2d(1), x1(2), x2(2), x1d(2), x2d(2), ...]
Data = reshape(zeta, model.nbVarPos*model.nbDeriv, nbData*nbSamples); %Include derivatives in Data
for n=1:nbSamples
	s(n).Data = Data(:,(n-1)*nbData+1:n*nbData);
end

%% Parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Initialization with kmeans
% model = init_GMM_kmeans(Data, model);

%Initialization with equal sequential bins
model = init_GMM_kbins(Data, model, nbSamples);

%Refinement of parameters
model = EM_GMM(Data, model);

%Precomputation of inverses (optional)
for i=1:model.nbStates
	model.invSigma(:,:,i) = inv(model.Sigma(:,:,i));
end

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

model.params_updateComp

[model, H] = EM_HMM(s, model);
%Removal of self-transition (for HSMM representation) and normalization
model.Trans = model.Trans - diag(diag(model.Trans)) + eye(model.nbStates)*realmin;
model.Trans = model.Trans ./ repmat(sum(model.Trans,2),1,model.nbStates);


%% Post-estimation of the state duration from data (for HSMM representation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
		st(currState).d = [st(currState).d cnt];
		cnt = 1;
		currState = hmax(t);
	end
end
st(currState).d = [st(currState).d cnt];

%Compute state duration as Gaussian distribution (optional)
for i=1:model.nbStates
	model.Mu_Pd(1,i) = mean(st(i).d);
	model.Sigma_Pd(1,1,i) = cov(st(i).d) + minSigmaPd;
end


%% Reconstruction of states probability sequence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbD = round(2 * nbData/model.nbStates); %Number of maximum duration step to consider in the HSMM (2 is a safety factor)

%Precomputation of duration probabilities 
for i=1:model.nbStates
	model.Pd(i,:) = gaussPDF([1:nbD], model.Mu_Pd(:,i), model.Sigma_Pd(:,:,i)); 
	%The rescaling formula below can be used to guarantee that the cumulated sum is one (to avoid the numerical issues)
	model.Pd(i,:) = model.Pd(i,:) / sum(model.Pd(i,:));
end

%Reconstruction of states sequence based on standard computation
%(in the iteration, a scaling factor c is used to avoid numerical underflow issues in HSMM, see Levinson'1986) 
h = zeros(model.nbStates,nbData);
c = zeros(nbData,1); %scaling factor to avoid numerical issues
c(1)=1; %Initialization of scaling factor
for t=1:nbData
	for i=1:model.nbStates
		if t<=nbD
			oTmp = 1; %Observation probability for generative purpose
% 			oTmp = prod(c(1:t) .* gaussPDF(s(1).Data(:,1:t), model.Mu(:,i), model.Sigma(:,:,i))); %Observation probability for standard HSMM
			h(i,t) = model.StatesPriors(i) * model.Pd(i,t) * oTmp;
		end
		for d=1:min(t-1,nbD)
			oTmp = 1; %Observation probability for generative purpose
% 			oTmp = prod(c(t-d+1:t) .* gaussPDF(s(1).Data(:,t-d+1:t), model.Mu(:,i), model.Sigma(:,:,i))); %Observation probability for standard HSMM	
			h(i,t) = h(i,t) + h(:,t-d)' * model.Trans(:,i) * model.Pd(i,d) * oTmp;
		end
	end
	c(t+1) = 1/sum(h(:,t)); %Update of scaling factor
end
h = h ./ repmat(sum(h,1),model.nbStates,1);


%% Reconstruction of trajectory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Compute best path for the n-th demonstration
[~,r(1).q] = max(h,[],1); %works also for nbStates=1

%Create single Gaussian N(MuQ,SigmaQ) based on optimal state sequence q
MuQ = reshape(model.Mu(:,r(1).q), model.nbVar*nbData, 1); 
%MuQ = zeros(model.nbVar*nbData,1);

SigmaQ = (kron(ones(nbData,1), eye(model.nbVar)) * reshape(model.Sigma(:,:,r(1).q), model.nbVar, model.nbVar*nbData)) .* kron(eye(nbData), ones(model.nbVar));

% SigmaQ = zeros(model.nbVar*nbData);
% for t=1:nbData
% 	id = (t-1)*model.nbVar+1:t*model.nbVar;
% 	%MuQ(id) = model.Mu(:,r(1).q(t)); 
% 	SigmaQ(id,id) = model.Sigma(:,:,r(1).q(t));
% end

%Least squares computation method 1 using lscov Matlab function (with Octave, use method 2 below)
[xhat,~,~,S] = lscov(PHI1, MuQ, SigmaQ, 'chol'); %Retrieval of data with weighted least squares solution
r(1).Data = reshape(xhat, model.nbVarPos, nbData); %Reshape data for plotting

% %Least squares computation method 2 (most readable but not optimized)
% PHIinvSigmaQ = PHI1' / SigmaQ;
% Rq = PHIinvSigmaQ * PHI1;
% rq = PHIinvSigmaQ * MuQ;
% xhat = Rq \ rq; %Can also be computed with c = lscov(Rq, rq)
% size(zeta)
% r(1).Data = reshape(xhat, model.nbVarPos, nbData); %Reshape data for plotting
% %Covariance Matrix computation of ordinary least squares estimate
% mse =  (MuQ'*inv(SigmaQ)*MuQ - rq'*inv(Rq)*rq) ./ ((model.nbVar-model.nbVarPos)*nbData);
% S = inv(Rq) * mse; 

%Rebuild covariance by reshaping S
for t=1:nbData
	id = (t-1)*model.nbVarPos+1:t*model.nbVarPos;
	r(1).expSigma(:,:,t) = S(id,id) * nbData;
end


%% Plot timeline
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 600 600]);
for m=1:model.nbVarPos
	limAxes = [1, nbData, min(Data(m,:))-1E0, max(Data(m,:))+1E0];
	subplot(model.nbVarPos,1,m); hold on;
	msh=[]; x0=[];
	for t=1:nbData-1
		if size(msh,2)==0
			msh(:,1) = [t; model.Mu(m,r(1).q(t))];
		end
		if t==nbData-1 || r(1).q(t+1)~=r(1).q(t)
			msh(:,2) = [t+1; model.Mu(m,r(1).q(t))];
			sTmp = model.Sigma(m,m,r(1).q(t))^.5;
			msh2 = [msh(:,1)+[0;sTmp], msh(:,2)+[0;sTmp], msh(:,2)-[0;sTmp], msh(:,1)-[0;sTmp], msh(:,1)+[0;sTmp]];
			patch(msh2(1,:), msh2(2,:), [.85 .85 .85],'edgecolor',[.7 .7 .7]);
			plot(msh(1,:), msh(2,:), '-','linewidth',3,'color',[.7 .7 .7]);
			plot([msh(1,1) msh(1,1)], limAxes(3:4), ':','linewidth',1,'color',[.7 .7 .7]);
			x0 = [x0 msh];
			msh=[];
		end
	end
	msh = [1:nbData, nbData:-1:1; r(1).Data(m,:)-squeeze(r(1).expSigma(m,m,:).^.5)'*1, fliplr(r(1).Data(m,:)+squeeze(r(1).expSigma(m,m,:).^.5)'*1)];
	patch(msh(1,:), msh(2,:), ones(1,size(msh,2)), [1 .4 .4],'edgecolor',[1 .2 .2],'edgealpha',.8,'facealpha',.5);
	for n=1:nbSamples
		plot(1:nbData, Data(m,(n-1)*nbData+1:n*nbData), '-','lineWidth',1,'color',[.2 .2 .2]);
	end
	%plot(1:nbData, model.Mu(m,r(1).q), '-','lineWidth',3.5,'color',[.8 0 0]);
	plot(1:nbData, r(1).Data(m,:), '-','lineWidth',3.5,'color',[.8 0 0]);
	set(gca,'xtick',[],'ytick',[]);
	xlabel('$t$','interpreter','latex','fontsize',18);
	ylabel(['$x_' num2str(m) '$'],'interpreter','latex','fontsize',18);
	axis(limAxes);
end
%print('-dpng','graphs/demo_trajHSMM01a.png');


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[620 10 600 600]); hold on;
plotGMM(model.Mu([1,2],:), model.Sigma([1,2],[1,2],:), [.5 .5 .5],.8);
plotGMM(r(1).Data([1,2],:), r(1).expSigma([1,2],[1,2],:), [1 .2 .2],.1);
for n=1:nbSamples
	plot(Data(1,(n-1)*nbData+1:n*nbData), Data(2,(n-1)*nbData+1:n*nbData), '-','lineWidth',1,'color',[.2 .2 .2]); 
end
%plot(model.Mu(1,r(1).q), model.Mu(2,r(1).q), '-','lineWidth',3.5,'color',[.8 0 0]);
plot(r(1).Data(1,:), r(1).Data(2,:), '-','lineWidth',3.5,'color',[.8 0 0]);
set(gca,'xtick',[],'ytick',[]); axis equal; axis square;
xlabel(['$x_1$'],'interpreter','latex','fontsize',18);
ylabel(['$x_2$'],'interpreter','latex','fontsize',18);
%print('-dpng','graphs/demo_trajHSMM01b.png');

pause;
close all;