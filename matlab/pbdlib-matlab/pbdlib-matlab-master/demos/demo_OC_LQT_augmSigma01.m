function demo_OC_LQT_augmSigma01
% Batch LQR with augmented covariance to transform the tracking problem to a regulation problem.
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
% Copyright (c) 2016 Idiap Research Institute, http://idiap.ch/
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
model.nbStates = 4; %Number of Gaussians in the GMM
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 2; %Number of static & dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.dt = 0.01; %Time step duration
model.rfactor = model.dt^model.nbDeriv;	%Control cost in LQR
nbSamples = 5; %Number of demonstrations
nbRepros = 1; %Number of reproductions in new situations
nbData = 200; %Number of datapoints

%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;
R = kron(eye(nbData-1),R);


%% Dynamical System settings (discrete version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %Integration with Euler method 
% Ac1d = diag(ones(model.nbDeriv-1,1),1); 
% Bc1d = [zeros(model.nbDeriv-1,1); 1];
% A0 = kron(eye(model.nbDeriv)+Ac1d*model.dt, eye(model.nbVarPos)); 
% B0 = kron(Bc1d*model.dt, eye(model.nbVarPos));

%Integration with higher order Taylor series expansion
A1d = zeros(model.nbDeriv);
for i=0:model.nbDeriv-1
	A1d = A1d + diag(ones(model.nbDeriv-i,1),i) * model.dt^i * 1/factorial(i); %Discrete 1D
end
B1d = zeros(model.nbDeriv,1); 
for i=1:model.nbDeriv
	B1d(model.nbDeriv-i+1) = model.dt^i * 1/factorial(i); %Discrete 1D
end
A0 = kron(A1d, eye(model.nbVarPos)); %Discrete nD
B0 = kron(B1d, eye(model.nbVarPos)); %Discrete nD

% %Conversion with control toolbox
% Ac1d = diag(ones(model.nbDeriv-1,1),1); %Continuous 1D
% Bc1d = [zeros(model.nbDeriv-1,1); 1]; %Continuous 1D
% Cc1d = [1, zeros(1,model.nbDeriv-1)]; %Continuous 1D
% sysd = c2d(ss(Ac1d,Bc1d,Cc1d,0), model.dt);
% A0 = kron(sysd.a, eye(model.nbVarPos)); %Discrete nD
% B0 = kron(sysd.b, eye(model.nbVarPos)); %Discrete nD

A = [A0, zeros(model.nbVar,1); zeros(1,model.nbVar), 1]; %Augmented A
B = [B0; zeros(1,model.nbVarPos)]; %Augmented B
%Build Sx and Su matrices for batch LQR
Su = zeros((model.nbVar+1)*nbData, (model.nbVarPos)*(nbData-1));
Sx = kron(ones(nbData,1),eye(model.nbVar+1));
M = B;
for n=2:nbData
	id1 = (n-1)*(model.nbVar+1)+1:nbData*(model.nbVar+1); 
	Sx(id1,:) = Sx(id1,:) * A;
	id1 = (n-1)*(model.nbVar+1)+1:n*(model.nbVar+1); 
	id2 = 1:(n-1)*model.nbVarPos;
	Su(id1,id2) = M;
	M = [A*M(:,1:model.nbVarPos), M]; %Also M = [A^(n-1)*B, M] or M = [Sx(id1,:)*B, M]
end


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data/2Dletters/A.mat');
x=[];
for n=1:nbSamples
	s(n).x=[];
	for m=1:model.nbDeriv
		if m==1
			dTmp = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
		else
			dTmp = gradient(dTmp) / model.dt; %Compute derivatives
		end
		s(n).x = [s(n).x; dTmp];
	end
	x = [x, s(n).x]; 
end


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Parameters estimation...');
%model = init_GMM_kmeans(x, model);
model = init_GMM_kbins(x, model, nbSamples);

% %Initialization based on position data
% model0 = init_GMM_kmeans(x(1:model.nbVarPos,:), model);
% [~, GAMMA2] = EM_GMM(x(1:model.nbVarPos,:), model0);
% model.Priors = model0.Priors;
% for i=1:model.nbStates
% 	model.Mu(:,i) = x * GAMMA2(i,:)';
% 	xTmp = x - repmat(model.Mu(:,i), 1, nbData*nbSamples);
% 	model.Sigma(:,:,i) = xTmp * diag(GAMMA2(i,:)) * xTmp';
% end

%Refinement of parameters
[model, H] = EM_GMM(x, model);

%Transform model to the corresponding version with augmented covariance
model0 = model;
model.nbVar = model0.nbVar+1;
model.Mu = zeros(model.nbVar, model.nbStates);
model.Sigma = zeros(model.nbVar, model.nbVar, model.nbStates);
for i=1:model.nbStates
% 	model.Sigma(:,:,i) = [model0.Sigma(:,:,i)+model0.Mu(:,i)*model0.Mu(:,i)', model0.Mu(:,i); model0.Mu(:,i)', 1];
	model.Sigma(:,:,i) = [model0.Sigma(:,:,i)+model0.Mu(:,i)*model0.Mu(:,i)', model0.Mu(:,i); model0.Mu(:,i)', 1] .* (det(model0.Sigma(:,:,i)).^(-1./(model.nbVar+1)));
end


%% Batch LQR reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,qList] = max(H(:,1:nbData),[],1); %works also for nbStates=1
SigmaQ = (kron(ones(nbData,1), eye(model.nbVar)) * reshape(model.Sigma(:,:,qList), model.nbVar, (model.nbVar)*nbData)) .* kron(eye(nbData), ones(model.nbVar));

%Set matrices to compute the damped weighted least squares estimate, see Eq. (37)
SuInvSigmaQ = Su' / SigmaQ;
Rq = SuInvSigmaQ * Su + R;

%Reproductions
for n=1:nbRepros
	if n==1
		X = [x(:,1); 1];
	else
		X = [x(:,1); 1] + [randn(model.nbVarPos,1)*2E0; zeros(model.nbVar-model.nbVarPos,1)];
	end
	rq = -SuInvSigmaQ * Sx * X;
 	u = Rq \ rq; %Can also be computed with u = lscov(Rq, rq);
	r(n).x = reshape(Sx*X+Su*u, model.nbVar, nbData);
end


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 700 650],'color',[1 1 1]); hold on; axis off;
plotGMM(model0.Mu(1:2,:), model0.Sigma(1:2,1:2,:), [0.5 0.5 0.5]);
plot(x(1,:), x(2,:), 'k.');
for n=1:nbRepros
	plot(r(n).x(1,:), r(n).x(2,:), '-','linewidth',2,'color',[.8 0 0]);
end
axis equal; 


%% Timeline plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labList = {'$x_1$','$x_2$','$\dot{x}_1$','$\dot{x}_2$','$\ddot{x}_1$','$\ddot{x}_2$'};
figure('position',[720 10 600 650],'color',[1 1 1]); 
for j=1:model0.nbVar
subplot(model.nbVar+1,1,j); hold on;
for n=1:nbSamples
	plot(x(j,(n-1)*nbData+1:n*nbData), '-','linewidth',.5,'color',[0 0 0]);
end
for n=1:nbRepros
	plot(r(n).x(j,:), '-','linewidth',1,'color',[.8 0 0]);
end
if j<7
	ylabel(labList{j},'fontsize',14,'interpreter','latex');
end
end

%Speed profile
if model.nbDeriv>1
subplot(model.nbVar+1,1,model.nbVar+1); hold on;
for n=1:nbSamples
	sp = sqrt(x(3,(n-1)*nbData+1:n*nbData).^2 + x(4,(n-1)*nbData+1:n*nbData).^2);
	plot(sp, '-','linewidth',.5,'color',[0 0 0]);
end
for n=1:nbRepros
	sp = sqrt(r(n).x(3,:).^2 + r(n).x(4,:).^2);
	plot(sp, '-','linewidth',1,'color',[.8 0 0]);
end
ylabel('$|\dot{x}|$','fontsize',14,'interpreter','latex');
xlabel('$t$','fontsize',14,'interpreter','latex');
end

%print('-dpng','graphs/demo_batchLQR_augmSigma01.png');
pause;
close all;