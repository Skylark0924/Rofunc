function benchmark_DS_PGMM01
% Benchmark of task-parameterized model based on parametric Gaussian mixture model, 
% and DS-GMR used for reproduction.
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
model.nbVar = 3; %Dimension of the datapoints in the dataset (here: t,x1,x2)
model.dt = 0.01; %Time step
model.kP = 100; %Stiffness gain
model.kV = (2*model.kP)^.5; %Damping gain (with ideal underdamped damping ratio)
nbRepros = 4; %Number of reproductions with new situations randomly generated
nbVarOut = model.nbVar-1; %(here, x1,x2)
L = [eye(nbVarOut)*model.kP, eye(nbVarOut)*model.kV]; %Feedback gains


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


%% Transformation of 'Data' to learn the path of the spring-damper system
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbD = s(1).nbData;
%Create transformation matrix to compute [X; DX; DDX]
D = (diag(ones(1,nbD-1),-1)-eye(nbD)) / model.dt;
D(end,end) = 0;
%Create transformation matrix to compute XHAT = X + DX*kV/kP + DDX/kP
K1d = [1, model.kV/model.kP, 1/model.kP];
K = kron(K1d,eye(nbVarOut));
%Compute derivatives
Data = [];
for n=1:nbSamples
	DataTmp = s(n).Data0(2:end,:);
	s(n).Data = [s(n).Data0(1,:); K * [DataTmp; DataTmp*D; DataTmp*D*D]];
	Data = [Data s(n).Data];
end


%% PGMM learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Parameters estimation of PGMM with EM:');
for n=1:nbSamples	
% 	%Task parameters rearranged as a vector (position and orientation)
% 	s(n).OmegaMu = [s(n).p(1).b(2:3); s(n).p(1).A(2:3,3); s(n).p(2).b(2:3); s(n).p(2).A(2:3,3); 1];

	%Task parameters rearranged as a vector (position only)
	s(n).OmegaMu = [s(n).p(1).b(2:3); s(n).p(2).b(2:3); 1];
end

% %Initialization of model parameters
% %model = init_GMM_kmeans(Data, model);
% model = init_GMM_timeBased(Data, model);
% model = EM_GMM(Data, model);

for i=1:model.nbStates
% 	%Initialization of parameters based on standard GMM
% 	model.ZMu(:,:,i) = zeros(model.nbVar, size(s(1).OmegaMu,1));
% 	model.ZMu(:,end,i) = model.Mu(:,i);

		%Random initialization of parameters
		model.ZMu(:,:,i) = rand(model.nbVar,size(s(1).OmegaMu,1));
		model.Sigma(:,:,i) = eye(model.nbVar);
end
model.Priors = ones(model.nbStates) / model.nbStates; %Useful when using random initialization of parameters

%PGMM parameters estimation
model = EM_stdPGMM(s, model);


%% Reproduction with PGMM and DS-GMR for the task parameters used to train the model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Reproductions with DS-GMR...');
DataIn = [1:s(1).nbData] * model.dt;
nbVarOut = model.nbVar-1;
for n=1:nbSamples
	%Computation of the resulting Gaussians (for display purpose)
	for i=1:model.nbStates
		model.Mu(:,i) = model.ZMu(:,:,i) * s(n).OmegaMu; %Temporary Mu variable
	end
	r(n).Mu = model.Mu;
	
% 	%Retrieval of attractor path through GMR
% 	currTar = GMR(model, DataIn, 1, [2:model.nbVar]); 
% 	%Motion retrieval with spring-damper system
% 	x = s(n).p(1).b(2:model.nbVar);
% 	dx = zeros(nbVarOut,1);
% 	for t=1:s(n).nbData
% 		%Compute acceleration, velocity and position
% 		ddx =  -L * [x-currTar(:,t); dx]; 
% 		dx = dx + ddx * model.dt;
% 		x = x + dx * model.dt;
% 		r(n).Data(:,t) = x;
% 	end
end


%% Reproduction with PGMM and DS-GMR for new task parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('New reproductions with DS-GMR...');
load('./../data/taskParams.mat'); %Load new task parameters (new situation)
for n=1:nbRepros
	rnew(n).p = taskParams(n).p;
	
% 	%Task parameters re-arranged as a vector (position and orientation)
% 	rnew(n).OmegaMu = [rnew(n).p(1).b(2:3); rnew(n).p(1).A(2:3,3); rnew(n).p(2).b(2:3); rnew(n).p(2).A(2:3,3); 1];

	%Task parameters re-arranged as a vector (position only)
	rnew(n).OmegaMu = [rnew(n).p(1).b(2:3); rnew(n).p(2).b(2:3); 1];
	
	%Computation of the resulting Gaussians (for display purpose)
	for i=1:model.nbStates
		model.Mu(:,i) = model.ZMu(:,:,i) * rnew(n).OmegaMu; %Temporary Mu variable
	end
	rnew(n).Mu = model.Mu;
	%Retrieval of attractor path through GMR
	[rnew(n).currTar, rnew(n).currSigma] = GMR(model, DataIn, 1, [2:model.nbVar]); 
	%Motion retrieval with spring-damper system
	x = rnew(n).p(1).b(2:model.nbVar);
	dx = zeros(nbVarOut,1);
	for t=1:nbD
		%Compute acceleration, velocity and position
		ddx =  -L * [x-rnew(n).currTar(:,t); dx]; 
		dx = dx + ddx * model.dt;
		x = x + dx * model.dt;
		rnew(n).Data(:,t) = x;
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
	plotGMM(r(n).Mu(2:3,:),model.Sigma(2:3,2:3,:), [0 0 0], .04);
end
axis equal; axis(limAxes);
%print('-dpng','-r600','graphs/benchmark_DS_PGMM01.png');

%Plot reproductions in new situations
disp('[Press enter to see next reproduction attempt]');
h=[];
for n=1:nbRepros
	delete(h);
	h = plotPegs(rnew(n).p);
	h = [h plotGMM(rnew(n).currTar, rnew(n).currSigma,  [0 .8 0], .2)];
	h = [h plotGMM(rnew(n).Mu(2:3,:), model.Sigma(2:3,2:3,:),  myclr(3,:), .6)];
	h = [h patch([rnew(n).Data(1,:) rnew(n).Data(1,fliplr(1:nbD))], [rnew(n).Data(2,:) rnew(n).Data(2,fliplr(1:nbD))],...
		[1 1 1],'linewidth',1.5,'edgecolor',[0 0 0],'facealpha',0,'edgealpha',0.4)];
	h = [h plot(rnew(n).Data(1,1), rnew(n).Data(2,1),'.','markersize',12,'color',[0 0 0])];
	axis equal; axis(limAxes);
	%print('-dpng','-r600',['graphs/benchmark_DS_PGMM' num2str(n+1,'%.2d') '.png']);
	pause;
end

pause;
close all;
end

%Function to plot pegs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h = plotPegs(p, colPegs, fa)
	if ~exist('colPegs')
		colPegs = [0.2863 0.0392 0.2392; 0.9137 0.4980 0.0078];
		fa = 0.4;
	end
	pegMesh = [-4 -3.5; -4 10; -1.5 10; -1.5 -1; 1.5 -1; 1.5 10; 4 10; 4 -3.5; -4 -3.5]' *1E-1;
	for m=1:length(p)
		dispMesh = p(m).A(2:3,2:3) * pegMesh + repmat(p(m).b(2:3),1,size(pegMesh,2));
		h(m) = patch(dispMesh(1,:),dispMesh(2,:),colPegs(m,:),'linewidth',1,'edgecolor','none','facealpha',fa);
	end
end