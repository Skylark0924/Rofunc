function demo_GMR_DS01
% Gaussian mixture model (GMM), with Gaussian mixture regression(GMR) and dynamical systems 
% used for reproduction, with decay variable used as input (as in DMP).
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19chapter,
% 	author="Calinon, S. and Lee, D.",
% 	title="Learning Control",
% 	booktitle="Humanoid Robotics: a Reference",
% 	publisher="Springer",
% 	editor="Vadakkepat, P. and Goswami, A.", 
% 	year="2019",
% 	doi="10.1007/978-94-007-7194-9_68-1",
% 	pages="1--52"
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 5; %Number of states in the GMM
model.nbVar = 3; %Number of variables [t,x1,x2]
nbData = 200; %Length of each trajectory
model.dt = 0.01; %Time step
nbSamples = 5; %Number of demonstrations
model.kP = 50; %Stiffness gain
model.kV = (2*model.kP)^.5; %Damping gain (with ideal underdamped damping ratio)
model.alpha = 1.0; %Decay factor
nbVarOut = model.nbVar-1; %Dimension of spatial variables
L = [eye(nbVarOut)*model.kP, eye(nbVarOut)*model.kV]; %Feedback term


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sIn(1) = 1; %Initialization of decay term
for t=2:nbData
	sIn(t) = sIn(t-1) - model.alpha * sIn(t-1) * model.dt; %Update of decay term (ds/dt=-alpha s)
end
% %Create transformation matrix to compute xhat = x + dx*kV/kP + ddx/kP
% K1d = [1, model.kV/model.kP, 1/model.kP];
% K = kron(K1d,eye(nbVarOut));
% %Create transformation matrix to compute [X; DX; DDX]
% D = (diag(ones(1,nbData-1),-1)-eye(nbData)) / model.dt;
% D(end,end) = 0;

load('data/2Dletters/G.mat');
Data=[]; Data0=[];
for n=1:nbSamples
	DataTmp = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	%s(n).Data = K * [DataTmp; DataTmp*D; DataTmp*D*D];
	s(n).Data = DataTmp;
	Data = [Data [sIn; s(n).Data]]; %Training data as [s;xhat]
	Data0 = [Data0 [sIn; DataTmp]]; %Original data as [s;x]
end


%% Learning and reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%model = init_GMM_timeBased(Data, model);
model = init_GMM_kmeans(Data, model);
model = EM_GMM(Data, model);
[currTar, currSigma] = GMR(model, sIn, 1, 2:model.nbVar); %see Eq. (17)-(19)
%Motion retrieval with spring-damper system
x = Data(2:model.nbVar,1);
dx = zeros(nbVarOut,1);
for t=1:nbData
	%Compute acceleration, velocity and position
	ddx = L * [currTar(:,t)-x; -dx]; 
	dx = dx + ddx * model.dt;
	x = x + dx * model.dt;
	DataOut(:,t) = x;
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1000,500]); 
%Plot GMM
subplot(1,2,1); hold on; box on; title('GMM');
plotGMM(model.Mu(2:model.nbVar,:), model.Sigma(2:model.nbVar,2:model.nbVar,:), [.8 0 0]);
plot(Data0(2,:),Data0(3,:),'.','markersize',8,'color',[.7 .7 .7]);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);
%Plot DS-GMR
subplot(1,2,2); hold on; box on; title('DS-GMR');
plotGMM(currTar, currSigma, [0 .8 0]);
plot(Data0(2,:),Data0(3,:),'.','markersize',8,'color',[.7 .7 .7]);
plot(currTar(1,:),currTar(2,:),'-','linewidth',1.5,'color',[0 .6 0]);
plot(DataOut(1,:),DataOut(2,:),'-','linewidth',3,'color',[0 .3 0]);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);

%print('-dpng','graphs/demo_DSGMR01.png');
pause;
close all;