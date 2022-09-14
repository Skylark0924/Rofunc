function benchmark_DS_TP_MFA01
% Benchmark of task-parameterized mixture of factor analyzers (TP-MFA), 
% with DS-GMR used for reproduction.
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
model.nbFA = 1; %Dimension of factor analyzers
model.dt = 0.01; %Time step
model.kP = 100; %Stiffness gain
model.kV = (2*model.kP)^.5; %Damping gain (with ideal underdamped damping ratio)
nbRepros = 4; %Number of reproductions with new situations randomly generated
nbStochasticRepros = 30; %Number of reproductions with stochastic sampling


%% Load 3rd order tensor data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Load 3rd order tensor data...');
% The MAT file contains a structure 's' with the multiple demonstrations. 's(n).Data' is a matrix data for
% sample n (with 's(n).nbData' datapoints). 's(n).p(m).b' and 's(n).p(m).A' contain the position and
% orientation of the m-th candidate coordinate system for this demonstration. 'Data' contains the observations
% in the different frames. It is a 3rd order tensor of dimension D x P x N, with D=3 the dimension of a
% datapoint, P=2 the number of candidate frames, and N=200x4 the number of datapoints in a trajectory (200)
% multiplied by the number of demonstrations (5).
load('./../data/DataLQR01.mat');


%% Transformation of 'Data' to learn the path of the spring-damper system
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbD = s(1).nbData;
nbVarOut = model.nbVar - 1;
%Create transformation matrix to compute [X; DX; DDX]
D = (diag(ones(1,nbD-1),-1)-eye(nbD)) / model.dt;
D(end,end) = 0;
%Create transformation matrix to compute XHAT = X + DX*kV/kP + DDX/kP
K1d = [1, model.kV/model.kP, 1/model.kP];
K = kron(K1d,eye(nbVarOut));
%Create 3rd order tensor data with XHAT instead of X
Data = zeros(model.nbVar, model.nbFrames, nbD*nbSamples);
for n=1:nbSamples
	DataTmp = s(n).Data0(2:end,:);
	DataTmp = [s(n).Data0(1,:); K * [DataTmp; DataTmp*D; DataTmp*D*D]];
	for m=1:model.nbFrames
		Data(:,m,(n-1)*nbD+1:n*nbD) = s(n).p(m).A \ (DataTmp - repmat(s(n).p(m).b, 1, nbD));
	end
end


%% Tensor MFA learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Parameters estimation of tensor GMM with EM:');
%model = init_tensorGMM_kmeans(Data, model); %Initialization
model = init_tensorGMM_timeBased(Data, model); %Initialization
model = EM_tensorMFA(Data, model);


%% Reproduction for the task parameters used to train the model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Reproductions with DS-GMR...');
DataIn = [1:nbD] * model.dt;
for n=1:nbSamples
	%Retrieval of attractor path through task-parameterized GMR
	a(n) = estimateAttractorPath(DataIn, model, s(n));
	r(n) = reproduction_DS(DataIn, model, a(n), s(n).p(1).b(2:3));
end


%% Reproduction for new task parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('New reproductions with DS-GMR...');
load('./../data/taskParams.mat'); %Load new task parameters (new situation)
for n=1:nbRepros
	%Retrieval of attractor path through task-parameterized GMR
	anew(n) = estimateAttractorPath(DataIn, model, taskParams(n));
	rnew(n) = reproduction_DS(DataIn, model, anew(n), taskParams(n).p(1).b(2:3));
end


%% Reproduction with stochastic sampling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Stochastic sampling with DS-GMR...');
for n=1:nbStochasticRepros
	%Retrieval of attractor path through task-parameterized GMR
	mtmp = model;
	for i=1:mtmp.nbStates
		[V,D] = eig(mtmp.Sigma(:,:,i));
		mtmp.Mu(:,i) = mtmp.Mu(:,i) + V*D^.5 * 0.8 * randn(model.nbVar,1);
	end
	asto(n) = estimateAttractorPath(DataIn, mtmp, taskParams(3));
	rsto(n) = reproduction_DS(DataIn, mtmp, asto(n), taskParams(3).p(1).b(2:3));
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
	plotGMM(r(n).Mu(2:3,:),r(n).Sigma(2:3,2:3,:), [0 0 0], .04);
end
axis equal; axis(limAxes);
%print('-dpng','-r600','graphs/benchmark_DS_TP_MFA01.png');

% %Plot reproductions in new situations
% disp('[Press enter to see next reproduction attempt]');
% h=[];
% for n=1:nbRepros
% 	delete(h);
% 	h = plotPegs(rnew(n).p);
% 	h = [h plotGMM(rnew(n).currTar, anew(n).currSigma,  [0 .8 0], .1)];
% 	%h = [h plotGMM(rnew(n).Mu(2:3,:), rnew(n).Sigma(2:3,2:3,:),  myclr(3,:), .6)];
% 	h = [h patch([rnew(n).Data(2,:) rnew(n).Data(2,fliplr(1:nbD))], [rnew(n).Data(3,:) rnew(n).Data(3,fliplr(1:nbD))],...
% 		[1 1 1],'linewidth',1.5,'edgecolor',[0 0 0],'facealpha',0,'edgealpha',0.4)];
% 	h = [h plot(rnew(n).Data(2,1), rnew(n).Data(3,1),'.','markersize',12,'color',[0 0 0])];
% 	axis equal; axis(limAxes);
% 	%print('-dpng','-r600',['graphs/benchmark_DS_TP_MFA' num2str(n+1,'%.2d') '.png']);
% 	%pause;
% end

%Plot stochastic sampling in new situations
disp('[Press enter to see next reproduction attempt]');
h=[];
plotPegs(rsto(1).p);
for n=1:nbStochasticRepros
	patch([rsto(n).Data(2,:) rsto(n).Data(2,fliplr(1:nbD))], [rsto(n).Data(3,:) rsto(n).Data(3,fliplr(1:nbD))],...
		[1 1 1],'linewidth',1.5,'edgecolor',[0 0 0],'facealpha',0,'edgealpha',0.2);
	%plot(rsto(n).Data(2,1), rsto(n).Data(3,1),'.','markersize',12,'color',[0 0 0]);
end
axis equal; axis(limAxes);	
%print('-dpng','-r600',['graphs/benchmark_DS_TP_MFA_stochastic01.png']);
%pause;

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
		dispMesh = p(m).A(2:3,2:3) * pegMesh + repmat(p(m).b(2:3),1,size(pegMesh,2));
		h(m) = patch(dispMesh(1,:),dispMesh(2,:),colPegs(m,:),'linewidth',1,'edgecolor','none','facealpha',fa);
	end
end