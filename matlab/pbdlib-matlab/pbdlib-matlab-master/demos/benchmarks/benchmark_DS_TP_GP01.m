function benchmark_DS_TP_GP01
% Benchmark of task-parameterized Gaussian process (nonparametric task-parameterized method).
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
K = kron(K1d,eye(model.nbVar-1));
%Create 3rd order tensor data with XHAT instead of X
Data = zeros(model.nbVar, model.nbFrames, nbD*nbSamples);
for n=1:nbSamples
	DataTmp = s(n).Data0(2:end,:);
	s(n).Data = [s(n).Data0(1,:); K * [DataTmp; DataTmp*D; DataTmp*D*D]];
	for m=1:model.nbFrames
		Data(:,m,(n-1)*nbD+1:n*nbD) = s(n).p(m).A \ (s(n).Data - repmat(s(n).p(m).b, 1, nbD));
		s(n).p(m).A = s(n).p(m).A(2:end,2:end);
		s(n).p(m).b = s(n).p(m).b(2:end);
	end
end


%% Reproduction with TP-GP for the task parameters used to train the model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Reproductions with TP-GP and spring-damper system...');
in=1; out=2:model.nbVar;
model.nbVar = model.nbVar-1;
model.nbStates = nbD;
model.Priors = ones(model.nbStates,1) / model.nbStates;
for m=1:model.nbFrames
	DataIn(1,:) = squeeze(Data(in,m,:));
	DataOut = squeeze(Data(out,m,:));
	[MuTmp, SigmaTmp] = GPR(DataIn, DataOut, DataIn(1,1:nbD), [1E1, 1E2, 1E0]);
	model.Mu(:,m,:) = MuTmp;
	model.Sigma(:,:,m,:) = SigmaTmp;
end

%Reproduction with spring-damper system
% for n=1:nbSamples
% 	currTar = productTPGMM0(model, s(n).p); 
% 
% 	%Motion retrieval with spring-damper system
% 	x = s(n).p(1).b;
% 	dx = zeros(model.nbVar,1);
% 	for t=1:s(n).nbData
% 		%Compute acceleration, velocity and position
% 		ddx =  -L * [x-currTar(:,t); dx]; 
% 		dx = dx + ddx * model.dt;
% 		x = x + dx * model.dt;
% 		r(n).Data(:,t) = x;
% 	end
% end


%% Reproduction with TP-GP for new task parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('New reproductions with TP-GP and spring-damper system...');
load('./../data/taskParams.mat'); %Load new task parameters (new situation)
for n=1:nbRepros
	for m=1:model.nbFrames
		rnew(n).p(m).b = taskParams(n).p(m).b(2:end);
		rnew(n).p(m).A = taskParams(n).p(m).A(2:end,2:end);
	end
	[rnew(n).currTar, rnew(n).currSigma] = productTPGMM0(model, rnew(n).p); 

	%Motion retrieval with spring-damper system
	x = rnew(n).p(1).b;
	dx = zeros(model.nbVar,1);
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
axis equal; axis(limAxes);
%print('-dpng','-r600','graphs/benchmark_DS_TP_GP01.png');

%Plot reproductions in new situations
disp('[Press enter to see next reproduction attempt]');
h=[];
for n=1:nbRepros
	delete(h);
	h = plotPegs(rnew(n).p);
	h = [h plotGMM(rnew(n).currTar, rnew(n).currSigma,  [0 .8 0], .2)];
	h = [h patch([rnew(n).Data(1,:) rnew(n).Data(1,fliplr(1:nbD))], [rnew(n).Data(2,:) rnew(n).Data(2,fliplr(1:nbD))],...
		[1 1 1],'linewidth',1.5,'edgecolor',[0 0 0],'facealpha',0,'edgealpha',0.4)];
	h = [h plot(rnew(n).Data(1,1), rnew(n).Data(2,1),'.','markersize',12,'color',[0 0 0])];
	axis equal; axis(limAxes);
	%print('-dpng','-r600',['graphs/benchmark_DS_TP_GP' num2str(n+1,'%.2d') '.png']);
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