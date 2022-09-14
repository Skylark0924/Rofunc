function demo_stdPGMM01
% Parametric Gaussian mixture model (PGMM) used for task adaptation, with DS-GMR employed  to retrieve continuous movements. 
% The approach is inspired by Wilson and Bobick (1999), with an implementation applied to the special case of Gaussian mixture models (GMM).
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
% @article{Wilson99,
%   author="Wilson, A. D. and Bobick, A. F.",
%   title="Parametric Hidden {M}arkov Models for Gesture Recognition",
%   journal="{IEEE} Trans. on Pattern Analysis and Machine Intelligence",
%   year="1999",
%   volume="21",
%   number="9",
%   pages="884--900"
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

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 3; %Number of Gaussians in the GMM
model.nbFrames = 2; %Number of candidate frames of reference
model.nbVar = 3; %Dimension of the datapoints in the dataset (here: t,x1,x2)
model.dt = 0.01; %Time step
model.kP = 100; %Stiffness gain (for DS-GMR)
model.kV = (2*model.kP)^.5; %Damping gain (with ideal underdamped damping ratio)
nbRepros = 8; %Number of reproductions with new situations randomly generated


%% Load 3rd order tensor data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Load motion data...');
% The MAT file contains a structure 's' with the multiple demonstrations. 's(n).Data' is a matrix data for
% sample n (with 's(n).nbData' datapoints). 's(n).p(m).b' and 's(n).p(m).A' contain the position and
% orientation of the m-th candidate coordinate system for this demonstration. 'Data' contains the observations
% in the different frames. It is a 3rd order tensor of dimension D x P x N, with D=3 the dimension of a
% datapoint, P=2 the number of candidate frames, and N=TM the number of datapoints in a trajectory (T=200)
% multiplied by the number of demonstrations (M=5).
load('data/DataLQR01.mat');


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
	%Task parameters rearranged as a vector (position and orientation)
	s(n).OmegaMu = [s(n).p(1).b(2:3); s(n).p(1).A(2:3,3); s(n).p(2).b(2:3); s(n).p(2).A(2:3,3); 1];
	
% 	%Task parameters rearranged as a vector (position only)
% 	s(n).OmegaMu = [s(n).p(1).b(2:3); s(n).p(2).b(2:3); 1];
end

%Initialization of model parameters
model = init_GMM_timeBased(Data, model);
model = EM_GMM(Data, model);
for i=1:model.nbStates
	%Initialization of parameters based on standard GMM
	model.ZMu(:,:,i) = zeros(model.nbVar, size(s(1).OmegaMu,1));
	model.ZMu(:,end,i) = model.Mu(:,i);
	
	% 	%Random initialization of parameters
	% 	model.ZMu(:,:,i) = rand(model.nbVar,size(s(1).OmegaMu,1));
	% 	model.Sigma(:,:,i) = eye(model.nbVar);
end
% model.Priors = ones(model.nbStates) / model.nbStates; %Useful when using random initialization of parameters

%PGMM parameters estimation
model = EM_stdPGMM(s, model);


%% Reproduction with PGMM and DS-GMR for the task parameters used to train the model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Reproductions with DS-GMR...');
DataIn = [1:s(1).nbData] * model.dt;
nbVarOut = model.nbVar-1;
L = [eye(nbVarOut)*model.kP, eye(nbVarOut)*model.kV];
for n=1:nbSamples
	%Computation of the resulting Gaussians (for display purpose)
	for i=1:model.nbStates
		model.Mu(:,i) = model.ZMu(:,:,i) * s(n).OmegaMu; %Temporary Mu variable
	end
	r(n).Mu = model.Mu;
	%Retrieval of attractor path through GMR
	currTar = GMR(model, DataIn, 1, [2:model.nbVar]); 
	%Motion retrieval with spring-damper system
	x = s(n).p(1).b(2:model.nbVar);
	dx = zeros(nbVarOut,1);
	for t=1:s(n).nbData
		%Compute acceleration, velocity and position
		ddx =  -L * [x-currTar(:,t); dx]; 
		dx = dx + ddx * model.dt;
		x = x + dx * model.dt;
		r(n).Data(:,t) = x;
	end
end


%% Reproduction with PGMM and DS-GMR for new task parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('New reproductions with DS-GMR...');
for n=1:nbRepros
	%Random generation of new task parameters
	for m=1:model.nbFrames
		id=ceil(rand(2,1)*nbSamples);
		w=rand(2); w=w/sum(w);
		rnew(n).p(m).b = s(id(1)).p(m).b * w(1) + s(id(2)).p(m).b * w(2);
		rnew(n).p(m).A = s(id(1)).p(m).A * w(1) + s(id(2)).p(m).A * w(2);
	end
	
	%Task parameters re-arranged as a vector (position and orientation)
	rnew(n).OmegaMu = [rnew(n).p(1).b(2:3); rnew(n).p(1).A(2:3,3); rnew(n).p(2).b(2:3); rnew(n).p(2).A(2:3,3); 1];
	
% 	%Task parameters re-arranged as a vector (position only)
% 	rnew(n).OmegaMu = [rnew(n).p(1).b(2:3); rnew(n).p(2).b(2:3); 1];
	
	%Computation of the resulting Gaussians (for display purpose)
	for i=1:model.nbStates
		model.Mu(:,i) = model.ZMu(:,:,i) * rnew(n).OmegaMu; %Temporary Mu variable
	end
	rnew(n).Mu = model.Mu;
	%Retrieval of attractor path through GMR
	currTar = GMR(model, DataIn, 1, [2:model.nbVar]); 
	%Motion retrieval with spring-damper system
	x = rnew(n).p(1).b(2:model.nbVar);
	dx = zeros(nbVarOut,1);
	for t=1:nbD
		%Compute acceleration, velocity and position
		ddx =  -L * [x-currTar(:,t); dx]; 
		dx = dx + ddx * model.dt;
		x = x + dx * model.dt;
		rnew(n).Data(:,t) = x;
	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,2300,800]);
xx = round(linspace(1,64,nbSamples));
clrmap = colormap('jet');
clrmap = min(clrmap(xx,:),.95);
limAxes = [-1.2 0.8 -1.1 0.9];
colPegs = [[.9,.5,.9];[.5,.9,.5]];

%DEMOS
subplot(1,3,1); hold on; box on; title('Demonstrations');
for n=1:nbSamples
	%Plot frames
	for m=1:model.nbFrames
		plot([s(n).p(m).b(2) s(n).p(m).b(2)+s(n).p(m).A(2,3)], [s(n).p(m).b(3) s(n).p(m).b(3)+s(n).p(m).A(3,3)], '-','linewidth',6,'color',colPegs(m,:));
		plot(s(n).p(m).b(2), s(n).p(m).b(3),'.','markersize',30,'color',colPegs(m,:)-[.05,.05,.05]);
	end
	%Plot trajectories
	plot(s(n).Data0(2,1), s(n).Data0(3,1),'.','markersize',12,'color',clrmap(n,:));
	plot(s(n).Data0(2,:), s(n).Data0(3,:),'-','linewidth',1.5,'color',clrmap(n,:));
end
axis(limAxes); axis square; set(gca,'xtick',[],'ytick',[]);

%REPROS
subplot(1,3,2); hold on; box on; title('Reproductions with PGMM');
for n=1:nbSamples
	%Plot frames
	for m=1:model.nbFrames
		plot([s(n).p(m).b(2) s(n).p(m).b(2)+s(n).p(m).A(2,3)], [s(n).p(m).b(3) s(n).p(m).b(3)+s(n).p(m).A(3,3)], '-','linewidth',6,'color',colPegs(m,:));
		plot(s(n).p(m).b(2), s(n).p(m).b(3),'.','markersize',30,'color',colPegs(m,:)-[.05,.05,.05]);
	end
end
for n=1:nbSamples
	%Plot trajectories
	plot(r(n).Data(1,1), r(n).Data(2,1),'.','markersize',12,'color',clrmap(n,:));
	plot(r(n).Data(1,:), r(n).Data(2,:),'-','linewidth',1.5,'color',clrmap(n,:));
end
for n=1:nbSamples
	%Plot Gaussians
	plotGMM(r(n).Mu(2:3,:), model.Sigma(2:3,2:3,:), [.5 .5 .5],.6);
end
axis(limAxes); axis square; set(gca,'xtick',[],'ytick',[]);

%NEW REPROS
subplot(1,3,3); hold on; box on; title('New reproductions with PGMM');
for n=1:nbRepros
	%Plot frames
	for m=1:model.nbFrames
		plot([rnew(n).p(m).b(2) rnew(n).p(m).b(2)+rnew(n).p(m).A(2,3)], [rnew(n).p(m).b(3) rnew(n).p(m).b(3)+rnew(n).p(m).A(3,3)], '-','linewidth',6,'color',colPegs(m,:));
		plot(rnew(n).p(m).b(2), rnew(n).p(m).b(3),'.','markersize',30,'color',colPegs(m,:)-[.05,.05,.05]);
	end
end
for n=1:nbRepros
	%Plot trajectories
	plot(rnew(n).Data(1,1), rnew(n).Data(2,1),'.','markersize',12,'color',[.2 .2 .2]);
	plot(rnew(n).Data(1,:), rnew(n).Data(2,:),'-','linewidth',1.5,'color',[.2 .2 .2]);
end
for n=1:nbRepros
	%Plot Gaussians
	plotGMM(rnew(n).Mu(2:3,:), model.Sigma(2:3,2:3,:), [.5 .5 .5],.6);
end
axis(limAxes); axis square; set(gca,'xtick',[],'ytick',[]);

%print('-dpng','graphs/demo_stdPGMM01.png');
pause;
close all;