function demo_OC_LQT_viapoints02
% Batch LQR with viapoints and a double integrator system, and an encoding of only position.
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbSamples = 5; %Number of demonstrations
nbRepros = 1; %Number of reproductions in new situations
nbData = 200; %Number of datapoints

model.nbStates = 8; %Number of Gaussians in the GMM
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 1; %Number of static & dynamic features (D=1 for just x)
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.dt = 0.01; %Time step duration
model.rfactor = 1E-8;	%Control cost in LQR

%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;
R = kron(eye(nbData-1),R);


%% Dynamical System settings (discrete version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbDeriv = model.nbDeriv + 1; %For definition of dynamical system

%Integration with higher order Taylor series expansion
A1d = zeros(nbDeriv);
for i=0:nbDeriv-1
	A1d = A1d + diag(ones(nbDeriv-i,1),i) * model.dt^i * 1/factorial(i); %Discrete 1D
end
B1d = zeros(nbDeriv,1); 
for i=1:nbDeriv
	B1d(nbDeriv-i+1) = model.dt^i * 1/factorial(i); %Discrete 1D
end
A = kron(A1d, eye(model.nbVarPos)); %Discrete nD
B = kron(B1d, eye(model.nbVarPos)); %Discrete nD
C = kron([1, 0], eye(model.nbVarPos));

%Build CSx and CSu matrices for batch LQR, see Eq. (35)
CSu = zeros(model.nbVarPos*nbData, model.nbVarPos*(nbData-1));
CSx = kron(ones(nbData,1), [eye(model.nbVarPos) zeros(model.nbVarPos)]);
M = B;
for n=2:nbData
	id1 = (n-1)*model.nbVarPos+1:n*model.nbVarPos;
	CSx(id1,:) = CSx(id1,:) * A;
	id1 = (n-1)*model.nbVarPos+1:n*model.nbVarPos; 
	id2 = 1:(n-1)*model.nbVarPos;
	CSu(id1,id2) = C * M;
	M = [A*M(:,1:model.nbVarPos), M]; %Also M = [A^(n-1)*B, M] or M = [Sx(id1,:)*B, M]
end


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data/2Dletters/M.mat');
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
	Data = [Data s(n).Data]; 
end


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Parameters estimation...');
%model = init_GMM_kmeans(Data,model);
model = init_GMM_kbins(Data,model,nbSamples);

% %Refinement of parameters
% [model, H] = EM_GMM(Data, model);

%Compute activation
MuRBF(1,:) = linspace(1, nbData, model.nbStates);
SigmaRBF = 1E2; 
H = zeros(model.nbStates,nbData);
for i=1:model.nbStates
	H(i,:) = gaussPDF(1:nbData, MuRBF(:,i), SigmaRBF);
end
H = H ./ repmat(sum(H,1),model.nbStates,1);

%Debug
model.Mu = (rand(model.nbVarPos,model.nbStates)-0.5).*2E1;
model.Sigma = repmat(eye(model.nbVarPos).*1E-1, [1,1,model.nbStates]);

%Precomputation of inverses
for i=1:model.nbStates
	model.invSigma(:,:,i) = inv(model.Sigma(:,:,i));
end

%Set list of states according to first demonstration (alternatively, an HSMM can be used)
[~,qList] = max(H(:,1:nbData),[],1); %works also for nbStates=1

%Create single Gaussian N(MuQ,SigmaQ) based on optimal state sequence q
MuQ = zeros(model.nbVar*nbData, 1); 
Q = zeros(model.nbVar*nbData);
qCurr = qList(1);
xDes = [];
for t=1:nbData
	if qCurr~=qList(t) || t==nbData
		id = (t-1)*model.nbVar+1:t*model.nbVar;
		MuQ(id) = model.Mu(:,qCurr); 
		Q(id,id) = model.invSigma(:,:,qCurr);
		%List of viapoints with time and error information (for plots)
		xDes = [xDes, [t; model.Mu(:,qCurr); diag(model.Sigma(:,:,qCurr)).^.5]];	
		qCurr = qList(t);
	end
end	


%% Batch LQR reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set matrices to compute the damped weighted least squares estimate
SuQ = CSu' * Q;
Rq = SuQ * CSu + R;
%Reproductions
for n=1:nbRepros
	X = [Data(:,1)+randn(model.nbVarPos,1)*0E0; zeros(model.nbVarPos,1)]; 
	rq = SuQ * (MuQ-CSx*X);
	u = Rq \ rq; %Can also be computed with u = lscov(Rq, rq);
	r(n).Data = reshape(CSx*X+CSu*u, model.nbVarPos, nbData);
end

%Compute velocities for visualization purpose
for n=1:nbSamples
	dTmp = gradient(s(n).Data) / model.dt; %Compute derivatives
	s(n).Data = [s(n).Data; dTmp];	
end
for n=1:nbRepros
	dTmp = gradient(r(n).Data) / model.dt; %Compute derivatives
	r(n).Data = [r(n).Data; dTmp];	
end


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 700 700],'color',[1 1 1],'name','x1-x2 plot'); hold on; axis off;
plotGMM(model.Mu(1:2,:), model.Sigma(1:2,1:2,:), [.5 .5 .5], .5);
plot(model.Mu(1,:), model.Mu(2,:), '.','markersize',15,'color',[.5 .5 .5]);
for n=1:nbRepros
	plot(r(n).Data(1,:), r(n).Data(2,:), '-','linewidth',2,'color',[.8 0 0]);
end
axis equal; 

figure('position',[10 610 700 700],'color',[1 1 1],'name','x1-dx1 plot'); hold on; axis off;
for n=1:nbRepros
	plot(r(n).Data(1,:), r(n).Data(3,:), '-','linewidth',2,'color',[.8 0 0]);
end
plot([min(Data(1,:)),max(Data(1,:))], [0,0], 'k:');


%% Timeline plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labList = {'$x_1$','$x_2$'};
figure('position',[710 10 1000 1300],'color',[1 1 1]); 
for j=1:model.nbVar
	subplot(model.nbVar+1,1,j); hold on;
	for n=1:nbRepros
		plot(r(n).Data(j,:), '-','linewidth',1,'color',[.8 0 0]);
	end
	for t=1:size(xDes,2)
		errorbar(xDes(1,t), xDes(1+j,t), xDes(1+model.nbVar+j,t), 'color',[.5 .5 .5]);
		plot(xDes(1,t), xDes(1+j,t), '.','markersize',15,'color',[.5 .5 .5]);
	end
	ylabel(labList{j},'fontsize',14,'interpreter','latex');
end
xlabel('$t$','fontsize',14,'interpreter','latex');
%Speed profile
subplot(model.nbVar+1,1,model.nbVar+1); hold on;
for n=1:nbSamples
	sp = sqrt(s(n).Data(3,:).^2 + s(n).Data(4,:).^2);
	plot(sp, '-','linewidth',.5,'color',[.6 .6 .6]);
end
for n=1:nbRepros
	sp = sqrt(r(n).Data(3,:).^2 + r(n).Data(4,:).^2);
	plot(sp, '-','linewidth',2,'color',[.8 0 0]);
end
plot(xDes(1,:), zeros(size(xDes,2),1), '.','markersize',15,'color',[.5 .5 .5]);
ylabel('$|\dot{x}|$','fontsize',14,'interpreter','latex');
xlabel('$t$','fontsize',14,'interpreter','latex');


%print('-dpng','graphs/demo_batchLQR_viapoints02.png');
pause;
close all;