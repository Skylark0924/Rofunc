function benchmark_DS_TP_trajGMM01
% Benchmark of task-parameterized Gaussian mixture model (TP-GMM), 
% with DS used for reproduction.
%
% If this code is useful for your research, please cite the related publication:
% @article{Calinon16JIST,
%   author="Calinon, S.",
%   title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
%   journal="Intelligent Service Robotics",
%		publisher="Springer Berlin Heidelberg",
%		doi="10.1007/s11370-015-0187-9",
%		year="2016",
%		volume="9",
%		number="1",
%		pages="1--29"
% }
% 
% Copyright (c) 2015 Idiap Research Institute, http://idiap.ch/
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

addpath('./../m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 3; %Number of Gaussians in the GMM
model.nbFrames = 2; %Number of candidate frames of reference
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 3; %Number of static&dynamic features (D=2 for [x,dx], D=3 for [x,dx,ddx], etc.)
model.dt = 0.1; %Time step
model.kP = 10; %Stiffness gain
model.kV = (2*model.kP)^.5; %Damping gain (with ideal underdamped damping ratio)
nbRepros = 4; %Number of reproductions with new situations randomly generated
L = [eye(model.nbVarPos)*model.kP, eye(model.nbVarPos)*model.kV]; %Feedback gains


%% Load 3rd order tensor data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Load 3rd order tensor data...');
% The MAT file contains a structure 's' with the multiple demonstrations. 's(n).Data' is a matrix data for
% sample n (with 's(n).nbData' datapoints). 's(n).p(m).b' and 's(n).p(m).A' contain the position and
% orientation of the m-th candidate coordinate system for this demonstration. 'Data' contains the observations
% in the different frames. It is a 3rd order tensor of dimension D x P x N, with D=3 the dimension of a
% datapoint, P=2 the number of candidate frames, and N=TM the number of datapoints in a trajectory (T=200)
% multiplied by the number of demonstrations (M=5).
load('./../data/DataLQR01.mat');


%% Transformation of Data to learn the path of the spring-damper system instead of the raw data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbD = s(1).nbData;
%Create transformation matrix to compute [X; DX; DDX]
D = (diag(ones(1,nbD-1),-1)-eye(nbD)) / model.dt;
D(end,end) = 0;
%Create transformation matrix to compute XHAT = X + DX*kV/kP + DDX/kP
K1d = [1, model.kV/model.kP, 1/model.kP];
K = kron(K1d,eye(model.nbVarPos));
%Create 3rd order tensor data with XHAT instead of X
for n=1:nbSamples
	DataTmp = s(n).Data0(2:end,:);
	s(n).Data = [s(n).Data0(1,:); K * [DataTmp; DataTmp*D; DataTmp*D*D]];
end


%% Compute derivatives
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbD = s(1).nbData;
%Create transformation matrix to compute [X; DX; DDX]
D = (diag(ones(1,nbD-1),-1)-eye(nbD)) / model.dt;
D(end,end) = 0;
%Create 3rd order tensor data with XHAT instead of X
model.nbVar = model.nbVarPos * model.nbDeriv;
Data = zeros(model.nbVar, model.nbFrames, nbD*nbSamples);
DataT = zeros(model.nbVar+1, model.nbFrames, nbD*nbSamples);
for n=1:nbSamples
	DataTmp = s(n).Data(2:end,:);
	for k=1:model.nbDeriv-1
		DataTmp = [DataTmp; s(n).Data(2:end,:)*D^k]; %Compute derivatives
	end
	for m=1:model.nbFrames
		s(n).p(m).b = [s(n).p(m).b(2:end); zeros((model.nbDeriv-1)*model.nbVarPos,1)];
		s(n).p(m).A = kron(eye(model.nbDeriv), s(n).p(m).A(2:end,2:end));
		Data(:,m,(n-1)*nbD+1:n*nbD) = s(n).p(m).A \ (DataTmp - repmat(s(n).p(m).b, 1, nbD));
		DataT(:,m,(n-1)*nbD+1:n*nbD) = [1:nbD; squeeze(Data(:,m,(n-1)*nbD+1:n*nbD))];
	end
end


%% Construct operator PHI (big sparse matrix)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T1 = nbD; %Number of datapoints in a demonstration
T = T1 * nbSamples; %Total number of datapoints
op1D = zeros(model.nbDeriv);
op1D(1,end) = 1;
for i=2:model.nbDeriv
	op1D(i,:) = (op1D(i-1,:) - circshift(op1D(i-1,:),[0,-1])) / model.dt;
end
op = zeros(T1*model.nbDeriv,T1);
op((model.nbDeriv-1)*model.nbDeriv+1:model.nbDeriv*model.nbDeriv,1:model.nbDeriv) = op1D;
PHI0 = zeros(T1*model.nbDeriv,T1);
for t=0:T1-model.nbDeriv
	PHI0 = PHI0 + circshift(op, [model.nbDeriv*t,t]);
end
%Handling of borders
for i=1:model.nbDeriv-1
	op(model.nbDeriv*model.nbDeriv+1-i,:)=0; op(:,i)=0;
	PHI0 = PHI0 + circshift(op, [-i*model.nbDeriv,-i]);
end
%Application to multiple dimensions and multiple demonstrations
PHI1 = kron(PHI0, eye(model.nbVarPos));
PHI = kron(eye(nbSamples), PHI1);


%% Tensor GMM learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Parameters estimation of tensor GMM with EM:');

% %k-means initialization
% model = init_tensorGMM_kmeans(Data, model); 

%Time-based initialization
modelT = model;
modelT.nbVar = model.nbVar+1;
modelT = init_tensorGMM_timeBased(DataT, modelT); 
for i=1:model.nbStates
	for m=1:model.nbFrames
		model.Mu(:,m,i) = modelT.Mu(2:end,m,i);
		model.Sigma(:,:,m,i) = modelT.Sigma(2:end,2:end,m,i);
	end
end
model.Priors = modelT.Priors;

model = EM_tensorGMM(Data, model);


%% Reproduction for the task parameters used to train the model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Reproductions...');
%DataIn = [1:s(1).nbData] * model.dt;
for n=1:nbSamples
	%Products of linearly transformed Gaussians
	for i=1:model.nbStates
		SigmaTmp = zeros(model.nbVar);
		MuTmp = zeros(model.nbVar,1);
		for m=1:model.nbFrames
			MuP = s(n).p(m).A * model.Mu(:,m,i) + s(n).p(m).b;
			SigmaP = s(n).p(m).A * model.Sigma(:,:,m,i) * s(n).p(m).A';
			SigmaTmp = SigmaTmp + inv(SigmaP);
			MuTmp = MuTmp + SigmaP\MuP;
		end
		r(n).Sigma(:,:,i) = inv(SigmaTmp);
		r(n).Mu(:,i) = r(n).Sigma(:,:,i) * MuTmp;
	end
end


%% Reproduction for new task parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('New reproductions...');
load('./../data/taskParams.mat'); %Load new task parameters (new situation)
for n=1:nbRepros
	%Adapt task parameters to trajectory GMM
	for m=1:model.nbFrames
		rnew(n).p(m).b = [taskParams(n).p(m).b(2:end); zeros((model.nbDeriv-1)*model.nbVarPos,1)];
		rnew(n).p(m).A = kron(eye(model.nbDeriv), taskParams(n).p(m).A(2:end,2:end));
	end
	%GMM products
	for i=1:model.nbStates
		SigmaTmp = zeros(model.nbVar);
		MuTmp = zeros(model.nbVar,1);
		for m=1:model.nbFrames
			MuP = rnew(n).p(m).A * model.Mu(:,m,i) + rnew(n).p(m).b;
			SigmaP = rnew(n).p(m).A * model.Sigma(:,:,m,i) * rnew(n).p(m).A';
			SigmaTmp = SigmaTmp + inv(SigmaP);
			MuTmp = MuTmp + SigmaP\MuP;
		end
		rnew(n).Sigma(:,:,i) = inv(SigmaTmp);
		rnew(n).Mu(:,i) = rnew(n).Sigma(:,:,i) * MuTmp;
	end
	%Create single Gaussian N(MuQ,SigmaQ) based on state sequence q
	%[~,rnew(n).q] = max(model.Pix(:,3*T1+1:4*T1),[],1); %works also for nbStates=1
	mPix = mean(reshape(model.Pix,[model.nbStates, nbD, nbSamples]),3); %Average Pix
	[~,rnew(n).q] = max(mPix,[],1); %works also for nbStates=1
	rnew(n).MuQ = reshape(rnew(n).Mu(:,rnew(n).q), model.nbVarPos*model.nbDeriv*T1, 1);
	rnew(n).SigmaQ = zeros(model.nbVarPos*model.nbDeriv*T1);
	for t=1:T1
		id1 = (t-1)*model.nbVarPos*model.nbDeriv+1:t*model.nbVarPos*model.nbDeriv;
		rnew(n).SigmaQ(id1,id1) = rnew(n).Sigma(:,:,rnew(n).q(t));
	end
	%Retrieval of data with weighted least squares solution
	[zeta,~,~,S] = lscov(PHI1, rnew(n).MuQ, rnew(n).SigmaQ,'chol');
	
	%Reproduction without DS
	%rnew(n).Data = reshape(zeta, model.nbVarPos, T1); %Reshape data
	
	%Reproduction with DS
	rnew(n).currTar = reshape(zeta, model.nbVarPos, T1); %Reshape data
	x = rnew(n).p(1).b(1:2);
	dx = zeros(model.nbVarPos,1);
	for t=1:nbD
		ddx =  -L * [x-rnew(n).currTar(:,t); dx];
		dx = dx + ddx * model.dt;
		x = x + dx * model.dt;
		rnew(n).Data(:,t) = x;
		%Rebuild covariance by reshaping S
		id = (t-1)*model.nbVarPos+1:t*model.nbVarPos;
		rnew(n).currSigma(:,:,t) = S(id,id) * nbD;
	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 4 3],'position',[20,50,600,450]);
axes('Position',[0 0 1 1]); axis off; hold on;
set(0,'DefaultAxesLooseInset',[0,0,0,0]);
limAxes = [-1.5 2.5 -1.6 1.4]*.8;
myclr = [0.2863 0.0392 0.2392; 0.9137 0.4980 0.0078; 0.7412 0.0824 0.3137];

%Plot demonstrations
plotPegs(s(1).p(1), myclr(1,:), .1);
for n=1:nbSamples
	plotPegs(s(n).p(2), myclr(2,:), .1);
	patch([s(n).Data0(2,1:end) s(n).Data0(2,end:-1:1)], [s(n).Data0(3,1:end) s(n).Data0(3,end:-1:1)],...
		[1 1 1],'linewidth',1.5,'edgecolor',[0 0 0],'facealpha',0,'edgealpha',0.04);
end
for n=1:nbSamples
	plotGMM(r(n).Mu(1:2,:),r(n).Sigma(1:2,1:2,:), [0 0 0], .04);
end
axis equal; axis(limAxes);
%print('-dpng','-r600','graphs/benchmark_DS_TP_trajGMM01.png');

%Plot reproductions in new situations
disp('[Press enter to see next reproduction attempt]');
h=[];
for n=1:nbRepros
	delete(h);
	h = plotPegs(rnew(n).p);
	h = [h plotGMM(rnew(n).currTar, rnew(n).currSigma,  [0 .8 0], .2)];
	h = [h plotGMM(rnew(n).Mu(1:2,:), rnew(n).Sigma(1:2,1:2,:),  myclr(3,:), .6)];
	h = [h patch([rnew(n).Data(1,:) rnew(n).Data(1,fliplr(1:nbD))], [rnew(n).Data(2,:) rnew(n).Data(2,fliplr(1:nbD))],...
		[1 1 1],'linewidth',1.5,'edgecolor',[0 0 0],'facealpha',0,'edgealpha',0.4)];
	h = [h plot(rnew(n).Data(1,1), rnew(n).Data(2,1),'.','markersize',12,'color',[0 0 0])];
	axis equal; axis(limAxes);
	%print('-dpng','-r600',['graphs/benchmark_DS_TP_trajGMM' num2str(n+1,'%.2d') '.png']);
	pause;
end

pause;
close all;
end

%Function to plot pegs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h = plotPegs(p, colPegs, fa)
	if ~exist('colPegs')
		colPegs = [0.2863    0.0392    0.2392; 0.9137    0.4980    0.0078];
		fa = 0.4;
	end
	pegMesh = [-4 -3.5; -4 10; -1.5 10; -1.5 -1; 1.5 -1; 1.5 10; 4 10; 4 -3.5; -4 -3.5]' *1E-1;
	for m=1:length(p)
		dispMesh = p(m).A(1:2,1:2) * pegMesh + repmat(p(m).b(1:2),1,size(pegMesh,2));
		h(m) = patch(dispMesh(1,:),dispMesh(2,:),colPegs(m,:),'linewidth',1,'edgecolor','none','facealpha',fa);
	end
end