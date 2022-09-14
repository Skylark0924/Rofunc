function demo_OC_LQT04
% Control of a spring attached to a point with batch LQR (with augmented state space)
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 6; %Number of Gaussians in the GMM
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 2; %Number of static & dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
model.nbVar = model.nbVarPos * (model.nbDeriv+1); %Dimension of augmented state vector
model.dt = 0.01; %Time step duration
model.rfactor = 1E-4;	%Control cost in LQR
model.kP = 50; %Stiffness gain (initial estimate)
model.kV = (2*model.kP)^.5; %Damping (with ideal damping ratio -> underdamped)
%model.kV = 2*model.kP^.5; %Damping (for critically damped system)
%model.xTar = [-1; -2]; %Equilibrium point of the spring

nbSamples = 5; %Number of demonstrations
nbRepros = 1; %Number of reproductions in new situations
nbData = 200; %Number of datapoints

% %Dynamical System settings (discrete version)
% Ac = kron([0, 1; -model.kP, -model.kV], eye(model.nbVarPos));
% A = eye(model.nbVar) + Ac * model.dt;
% B = kron([0; model.dt], eye(model.nbVarPos));

%Dynamical System settings (augmented state space, discrete version)
Ac = kron([0, 1, 0; -model.kP, -model.kV, model.kP; 0, 0, 0], eye(model.nbVarPos));
A = eye(model.nbVar) + Ac * model.dt;
B = kron([model.dt^2/2; model.dt; 0], eye(model.nbVarPos));

%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;
R = kron(eye(nbData-1),R);

%Build Sx and Su matrices for batch LQR
Su = zeros(model.nbVar*nbData, model.nbVarPos*(nbData-1));
Sx = kron(ones(nbData,1),eye(model.nbVar)); 
M = B;
for n=2:nbData
	id1 = (n-1)*model.nbVar+1:nbData*model.nbVar;
	Sx(id1,:) = Sx(id1,:) * A;
	id1 = (n-1)*model.nbVar+1:n*model.nbVar; 
	id2 = 1:(n-1)*model.nbVarPos;
	Su(id1,id2) = M;
	M = [A*M(:,1:model.nbVarPos), M];
end


%% Load handwriting data
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
	s(n).x = s(n).Data(1:model.nbVarPos,:);
	s(n).dx = s(n).Data(model.nbVarPos+1:end,:);
	s(n).ddx = gradient(dTmp) / model.dt;
	
	tarTmp = s(n).ddx ./ model.kP + s(n).dx .*model.kV./model.kP + s(n).x;
	Data = [Data [s(n).Data; tarTmp]];
	
	%Data = [Data [s(n).Data; repmat(s(n).Data(1:model.nbVarPos,end),1,nbData)]];
	%Data = [Data [s(n).Data; repmat(model.xTar,1,nbData)]]; 
end


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Parameters estimation...');
model = init_GMM_kbins(Data,model,nbSamples);
[model, H] = EM_GMM(Data, model);
[~,qList] = max(H(:,1:nbData),[],1); %works also for nbStates=1

%Precomputation of inverses 
for i=1:model.nbStates
% 	model.Mu(end-1:end,i)
% 	model.Mu(end-1:end,i) = rand(2,1) * 1E1;
% 	model.Mu(end-1:end,i)

	model.Mu(1:2,i) = model.Mu(end-1:end,i);
	%model.Mu(3:4,i) = 0;
	
	%model.invSigma(:,:,i) = inv(model.Sigma(:,:,i));
	%model.invSigma(:,:,i) = blkdiag(inv(model.Sigma(1:end-model.nbVarPos,1:end-model.nbVarPos,i)), zeros(model.nbVarPos)); %-> important to track observed paths 
	%model.invSigma(:,:,i) = blkdiag(zeros(model.nbVarPos*2), inv(model.Sigma(end-model.nbVarPos+1:end,end-model.nbVarPos+1:end,i))); %-> important to track virtual targets
	model.invSigma(:,:,i) = blkdiag(eye(model.nbVarPos)*1E2, eye(model.nbVarPos)*1E0, eye(model.nbVarPos)*0E-5); %-> important to track virtual targets
end


%% Batch LQR reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MuQ = reshape(model.Mu(:,qList), model.nbVar*nbData, 1); 
Q = (kron(ones(nbData,1), eye(model.nbVar)) * reshape(model.invSigma(:,:,qList), model.nbVar, model.nbVar*nbData)) .* kron(eye(nbData), ones(model.nbVar));
Q(1:12,1:12)
SuInvSigmaQ = Su' * Q;
Rq = SuInvSigmaQ * Su + R;
for n=1:nbRepros
	x = Data(:,1);
	%x = [Data(1:4,1); model.Mu(5:6,end)];
 	rq = SuInvSigmaQ * (MuQ-Sx*x);
 	u = Rq \ rq; 
	r(n).Data = reshape(Sx*x+Su*u, model.nbVar, nbData);
	r(n).u = reshape(u, model.nbVarPos, nbData-1);
end


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 1200 1200],'color',[1 1 1]); hold on; axis off;
colTmp = lines(model.nbStates);
for i=1:model.nbStates
	plotGMM(model.Mu(1:2,i), model.Sigma(1:2,1:2,i), colTmp(i,:), .2);
	%plotGMM(model.Mu(end-1:end,i), model.Sigma(end-1:end,end-1:end,i), colTmp(i,:), .05);
	plot(model.Mu(end-1,i), model.Mu(end,i), '.','markersize',20,'color',colTmp(i,:));
end
plot(Data(1,:), Data(2,:), '.', 'color',[.7 .7 .7]);
%plot(model.xTar(1),model.xTar(2),'k+','markersize',30,'linewidth',2);
for n=1:nbRepros
	plot(r(n).Data(1,:), r(n).Data(2,:), '-','linewidth',2,'color',[0 0 0]);
end
axis equal; 


%% Timeline plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labList = {'$x_1$','$x_2$','$\dot{x}_1$','$\dot{x}_2$','$\hat{x}_1$','$\hat{x}_2$'};
figure('position',[1220 10 1000 1200],'color',[1 1 1]); 
for j=1:model.nbVar
	subplot(model.nbVar,1,j); hold on; grid on;
	limAxes = [1, nbData, min(Data(j,:))-4E0, max(Data(j,:))+4E0];
	msh=[]; x0=[];
	for t=1:nbData-1
		if size(msh,2)==0
			msh(:,1) = [t; model.Mu(j,qList(t))];
		end
		if t==nbData-1 || qList(t+1)~=qList(t)
			i = qList(t);
			msh(:,2) = [t+1; model.Mu(j,i)];
			sTmp = model.Sigma(j,j,qList(t))^.5;
			msh2 = [msh(:,1)+[0;sTmp], msh(:,2)+[0;sTmp], msh(:,2)-[0;sTmp], msh(:,1)-[0;sTmp], msh(:,1)+[0;sTmp]];
			patch(msh2(1,:), msh2(2,:), colTmp(i,:),'edgecolor',colTmp(i,:),'facealpha', .4, 'edgealpha', .4);
			plot(msh(1,:), msh(2,:), '-','linewidth',1,'color',colTmp(i,:));
			if msh(1,1)>1
				plot([msh(1,1) msh(1,1)], limAxes(3:4), ':','linewidth',1,'color',[.7 .7 .7]);
			end
			x0 = [x0 msh];
			msh=[];
		end
	end
	if j<3
		plot(model.Mu(2*model.nbVarPos+j,qList),'b.');
	end
	if j<5
		for n=1:nbSamples
			plot(Data(j,(n-1)*nbData+1:n*nbData), '-','linewidth',.5,'color',[.6 .6 .6]);
		end
		for n=1:nbRepros
			plot(r(n).Data(j,:), '-','linewidth',2,'color',[0 0 0]);
		end		
	else
		for n=1:nbRepros
			plot(r(n).Data(j-4,:), '-','linewidth',2,'color',[0 0 0]);
		end
	end
	if j<7
		ylabel(labList{j},'fontsize',14,'interpreter','latex');
	end
	axis(limAxes);
end
xlabel('$t$','fontsize',14,'interpreter','latex');

%print('-dpng','graphs/demo_batchLQR04.png');
pause;
close all;